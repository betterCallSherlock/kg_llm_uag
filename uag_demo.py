"""
UaG on WebQSP

- Loads WebQSP_train.json and WebQSP_test.json (paths configurable)
- Builds a global KG by extracting triples (topic, inferential_chain_joined, answer)
- Runs UaG-style conformal predictor with Learn-Then-Test (LTT) lambda search
- Evaluates on test set and prints ECR / avg prediction set size and per-question details

Usage:
  pip install -r requirements.txt
  python uag_demo.py --train data/WebQSP_train.json --test data/WebQSP_test.json --max_train 500 --max_test 200

Notes:
- This is a runnable demo that uses embedding-similarity (all-MiniLM) for scoring.
- For speed, limit samples with --max_train / --max_test (defaults: use all) .
"""

import argparse
import json
import os
import random
import math
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import heapq

import networkx as nx
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------- helpers to parse WebQSP -----------------------------
def load_webqsp(path: str, max_samples: int = None) -> List[Dict]:
    """Load WebQSP JSON and return list of parsed items."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("Questions", []) if isinstance(data, dict) else data
    parsed = []
    for i, it in enumerate(items):
        if max_samples and i >= max_samples:
            break
        # Some WebQSP variants wrap questions under "Questions"; here each entry has 'ProcessedQuestion' and 'Parses'
        qid = it.get("QuestionId", it.get("id", f"q{i}"))
        proc_q = it.get("ProcessedQuestion") or it.get("RawQuestion") or ""
        # choose first parse (P0) by default
        parses = it.get("Parses", [])
        if not parses:
            continue
        parse = parses[0]
        topic = parse.get("TopicEntityName") or parse.get("PotentialTopicEntityMention") or None
        # answers may be multiple; use EntityName or AnswerArgument if EntityName missing
        answers = []
        for a in parse.get("Answers", []):
            en = a.get("EntityName")
            if en:
                answers.append(en)
            else:
                # fallback to AnswerArgument (may be mid or value)
                aa = a.get("AnswerArgument")
                if aa:
                    answers.append(str(aa))
        infer_chain = parse.get("InferentialChain") or []
        parsed.append({
            "id": qid,
            "question": proc_q,
            "topic": topic,
            "answers": answers,
            "infer_chain": infer_chain
        })
    return parsed

def build_global_triples(all_samples: List[Dict]) -> List[Tuple[str,str,str]]:
    """
    For each sample with topic + infer_chain + answer(s), create triples:
      (topic, relation_join, answer)
    relation_join is e.g. "film.actor.film->film.performance.character" if chain has two relations.
    Also, add simple relation triples (answer, 'is_answer_of', topic) for extra connectivity.
    """
    triples = []
    for s in all_samples:
        topic = s.get("topic")
        if not topic:
            continue
        chain = s.get("infer_chain", []) or []
        rel = "->".join(chain) if chain else "related_to"
        for ans in s.get("answers", []):
            if not ans:
                continue
            # primary directional triple
            triples.append((topic, rel, ans))
            # add reverse supporting triple
            triples.append((ans, "is_answer_of", topic))
    # deduplicate
    uniq = list({(h,r,t) for (h,r,t) in triples})
    return uniq

# ------------------------------- UaGCore (embedding-based) -------------------------------
class UaGCore:
    def __init__(self,
                 calibration_data: List[Dict],
                 global_triples: List[Tuple[str,str,str]],
                 path_alpha: float = 0.3,
                 ans_alpha: float = 0.2,
                 post_alpha: float = 0.1,
                 max_hop: int = 2,
                 encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        calibration_data: list of dicts with keys: question, topic (q_entity list), answers (a_entity list)
        global_triples: list of (head, relation, tail) used to build a single global KG graph
        """
        self.calibration_data = calibration_data
        self.path_alpha = path_alpha
        self.ans_alpha = ans_alpha
        self.post_alpha = post_alpha
        self.max_hop = max_hop
        self.encoder = SentenceTransformer(encoder_name)

        # build graph
        self.graph = self._build_graph(global_triples)

        # containers for calibration scores
        self.path_scores = defaultdict(list)
        self.ans_scores = []
        self.post_scores = []
        # thresholds
        self.q_hats = [0.0] * self.max_hop
        self.q_hat_a = 0.0
        self.q_hat_post = 0.0

        # compute calibration metrics and thresholds
        self._calculate_calibration_scores()
        self._compute_thresholds()

    def _build_graph(self, triples: List[Tuple[str,str,str]]):
        G = nx.DiGraph()
        for h, r, t in triples:
            G.add_edge(h, t, relation=r)
        return G

    def _embed(self, texts: List[str]):
        return self.encoder.encode(texts, convert_to_numpy=True)

    def _sim_score(self, s1: str, s2: str) -> float:
        # smaller => more similar
        embs = self._embed([s1, s2])
        sim = cosine_similarity(embs[:1], embs[1:])[0][0]
        return -sim

    @staticmethod
    def _mask_entities(question: str, entities: List[str]) -> str:
        s = question.lower()
        for e in entities:
            if not e:
                continue
            s = s.replace(e.lower().strip(), "[MASK]")
        return s

    def _calculate_calibration_scores(self):
        # For each calibration example, gather true shortest paths between topic and each answer (if exists),
        # and compute path-level and answer-level scores based on relations along those paths.
        for item in tqdm(self.calibration_data, desc="calc calib scores"):
            q = item["question"]
            qents = [item["topic"]] if item.get("topic") else []
            aents = item.get("answers", [])
            if not qents or not aents:
                continue
            G = self.graph
            masked_q = self._mask_entities(q, qents)
            # get shortest paths
            for qe in qents:
                for ae in aents:
                    if qe in G and ae in G:
                        try:
                            for path in nx.all_shortest_paths(G, qe, ae):
                                rels = []
                                # form triple sequence
                                for i in range(len(path)-1):
                                    u, v = path[i], path[i+1]
                                    rels.append(G[u][v]['relation'])
                                # compute per-hop scores
                                for hop_idx, rel in enumerate(rels):
                                    if hop_idx < self.max_hop:
                                        score = self._sim_score(masked_q + "?", rel)
                                        self.path_scores[hop_idx].append(score)
                                # answer-level: full path relation string
                                if rels:
                                    path_rel = " -> ".join(rels)
                                    score_a = self._sim_score(masked_q + "?", path_rel)
                                    self.ans_scores.append(score_a)
                                    score_post = self._sim_score(q, path_rel)
                                    self.post_scores.append(score_post)
                        except nx.NetworkXNoPath:
                            continue
        # ensure non-empty lists
        for hop in range(self.max_hop):
            if len(self.path_scores[hop]) == 0:
                self.path_scores[hop].append(0.0)
        if len(self.ans_scores) == 0:
            self.ans_scores.append(0.0)
        if len(self.post_scores) == 0:
            self.post_scores.append(0.0)

    def _compute_thresholds(self):
        # conformal quantiles per hop/answer/post
        self.q_hats = []
        for hop in range(self.max_hop):
            s = np.array(self.path_scores[hop])
            n = len(s)
            q = ((n + 1) * (1 - self.path_alpha)) / n
            q = min(max(q, 0.0), 1.0)
            self.q_hats.append(float(np.quantile(s, q)))
        s_a = np.array(self.ans_scores)
        n_a = len(s_a)
        q_a = ((n_a + 1) * (1 - self.ans_alpha)) / n_a
        q_a = min(max(q_a, 0.0), 1.0)
        self.q_hat_a = float(np.quantile(s_a, q_a))
        s_p = np.array(self.post_scores)
        n_p = len(s_p)
        q_p = ((n_p + 1) * (1 - self.post_alpha)) / n_p
        q_p = min(max(q_p, 0.0), 1.0)
        self.q_hat_post = float(np.quantile(s_p, q_p))

    def retrieve_candidates(self, q_entity_list: List[str], question: str):
        """Similar multi-hop retrieval as prior demos, but on global KG self.graph"""
        candidates = set()
        reasoning_paths = []
        path_conf = {}
        masked_q = self._mask_entities(question, q_entity_list)
        G = self.graph

        for q_ent in q_entity_list:
            if q_ent not in G:
                continue
            neighbors = list(G.neighbors(q_ent))
            queue = []
            for nb in neighbors:
                rel = G[q_ent][nb]['relation']
                score = self._sim_score(masked_q + "?", rel)
                if score <= self.q_hat_a:
                    candidates.add(nb)
                    formatted = f"{q_ent} -> {rel} -> {nb}"
                    reasoning_paths.append(formatted)
                    path_conf[formatted] = [score]
                if score <= self.q_hats[0]:
                    heapq.heappush(queue, (1, nb, [(q_ent, rel, nb)], [score]))
            while queue:
                depth, cur_ent, cur_path, cur_scores = heapq.heappop(queue)
                if depth >= self.max_hop:
                    continue
                parent = cur_path[-1][0]
                for nb2 in [n for n in G.neighbors(cur_ent) if n != parent]:
                    rel2 = G[cur_ent][nb2]['relation']
                    score2 = self._sim_score(masked_q + "? " + " -> ".join([r for _, r, _ in cur_path]), rel2)
                    new_scores = cur_scores + [score2]
                    new_path = cur_path + [(cur_ent, rel2, nb2)]
                    formatted = " -> ".join([f"{s} -> {r} -> {t}" for s, r, t in new_path])
                    path_full_rel = " -> ".join([r for _, r, _ in new_path])
                    ans_score = self._sim_score(masked_q + "?", path_full_rel)
                    if ans_score <= self.q_hat_a:
                        candidates.add(nb2)
                        reasoning_paths.append(formatted)
                        path_conf[formatted] = new_scores
                    hop_idx = depth
                    if hop_idx < len(self.q_hats) and score2 <= self.q_hats[hop_idx]:
                        if depth + 1 < self.max_hop:
                            heapq.heappush(queue, (depth + 1, nb2, new_path, new_scores))
        return candidates, reasoning_paths, path_conf

    def post_process(self, candidates: Set[str], reasoning_paths: List[str], question: str):
        """Select final answers using post-level threshold mapped to confidence."""
        if not candidates:
            return set(), {}
        ans_conf = {}
        for c in candidates:
            supporting = [p for p in reasoning_paths if p.lower().strip().endswith(c.lower().strip())]
            if not supporting:
                score = 10.0
            else:
                concat = " | ".join(supporting)
                score = self._sim_score(question, concat)
            scale = max(0.1, abs(self.q_hat_post) + 1e-6)
            conf = 1.0 / (1.0 + math.exp(score / scale))
            ans_conf[c] = conf
        threshold_conf = 1.0 / (1.0 + math.exp(self.q_hat_post / max(0.1, abs(self.q_hat_post))))
        selected = {c for c, conf in ans_conf.items() if conf >= threshold_conf}
        if not selected and ans_conf:
            selected = {max(ans_conf.items(), key=lambda x: x[1])[0]}
        return selected, ans_conf

    def predict(self, q_entity_list: List[str], question: str):
        candidates, paths, path_conf = self.retrieve_candidates(q_entity_list, question)
        final, per_conf = self.post_process(candidates, paths, question)
        # aggregate confidences
        mapped_path_conf = []
        for p, scores in path_conf.items():
            mapped = [1.0 / (1.0 + math.exp(s / (abs(self.q_hats[0]) + 1e-6))) for s in scores]
            if mapped:
                mapped_path_conf.append(float(np.mean(mapped)))
        avg_path_conf = float(np.mean(mapped_path_conf)) if mapped_path_conf else 0.0
        avg_ans_conf = float(np.mean(list(per_conf.values()))) if per_conf else 0.0
        return {
            "answers": final,
            "candidates": candidates,
            "paths": paths,
            "path_confidence": avg_path_conf,
            "answer_confidence": avg_ans_conf,
            "per_answer_conf": per_conf
        }

# ------------------------------- LTT lambda selection & evaluation -------------------------------
def split_calib(calib: List[Dict], seed: int = 42, val_frac: float = 0.5):
    rnd = random.Random(seed)
    data = calib.copy()
    rnd.shuffle(data)
    n_val = int(len(data) * val_frac)
    return data[n_val:], data[:n_val]

def evaluate_core_on_dataset_with_conf(core: UaGCore, dataset: List[Dict]):
    total = 0
    covered = 0
    sizes = []
    avg_confs = []
    details = []

    for item in dataset:
        total += 1
        qents = [item["topic"]] if item.get("topic") else []
        res = core.predict(qents, item["question"])

        # normalize answers
        pred = set([a.lower() for a in res["answers"]])
        true = set([a.lower() for a in item.get("answers", [])])
        confs = res.get("per_answer_conf", {})
        avg_conf = float(np.mean(list(confs.values()))) if confs else 0.0

        if len(pred & true) > 0:
            covered += 1

        sizes.append(len(pred))
        avg_confs.append(avg_conf)

        details.append({
            "id": item["id"],
            "question": item["question"],
            "true": list(true),
            "pred": list(pred),
            "covered": len(pred & true) > 0,
            "per_answer_conf": confs,
            "avg_conf": avg_conf
        })

    ecr = covered / total if total > 0 else 0.0
    avg_size = float(np.mean(sizes)) if sizes else 0.0
    mean_conf = float(np.mean(avg_confs)) if avg_confs else 0.0

    return {
        "ecr": ecr,
        "avg_size": avg_size,
        "mean_conf": mean_conf,
        "details": details
    }


def ltt_lambda_search(calib_data: List[Dict], global_triples: List[Tuple[str,str,str]],
                      lambda_grid: List[Tuple[float,float,float]], target_alpha: float, max_hop: int):
    train_calib, val_calib = split_calib(calib_data, seed=123, val_frac=0.5)
    stats = []
    valid = []
    for lam in lambda_grid:
        p_a, a_a, post_a = lam
        core = UaGCore(calibration_data=train_calib, global_triples=global_triples,
                       path_alpha=p_a, ans_alpha=a_a, post_alpha=post_a, max_hop=max_hop)
        res = evaluate_core_on_dataset_with_conf(core, val_calib)
        ok = res["ecr"] >= (1 - target_alpha)
        stats.append({"lambda": lam, "ecr": res["ecr"], "avg_size": res["avg_size"], "valid": ok})
        if ok:
            valid.append((lam, res["avg_size"]))
    if valid:
        chosen = min(valid, key=lambda x: x[1])[0]
    else:
        stats_sorted = sorted(stats, key=lambda x: (-x["ecr"], x["avg_size"]))
        chosen = stats_sorted[0]["lambda"]
    return chosen, stats

# ------------------------------- main pipeline -------------------------------
def build_calibration_items(parsed_train: List[Dict]):
    # adapt parsed_train entries to calibration format expected by UaGCore
    calib = []
    for it in parsed_train:
        if not it.get("topic") or not it.get("answers"):
            continue
        calib.append({
            "id": it["id"],
            "question": it["question"],
            "q_entity": [it["topic"]],
            "a_entity": it["answers"],
            # graph is global; UaGCore receives global_triples separately
            "topic": it["topic"],
            "answers": it["answers"]
        })
    return calib

def build_test_items(parsed_test: List[Dict]):
    tests = []
    for it in parsed_test:
        if not it.get("topic") or not it.get("answers"):
            continue
        tests.append({
            "id": it["id"],
            "question": it["question"],
            "q_entity": [it["topic"]],
            "a_entity": it["answers"],
            "topic": it["topic"],
            "answers": it["answers"]
        })
    return tests

def main(args):
    # load data
    print("Loading WebQSP files...")
    train_parsed = load_webqsp(args.train, max_samples=args.max_train)
    test_parsed = load_webqsp(args.test, max_samples=args.max_test)
    print(f"Loaded {len(train_parsed)} train items, {len(test_parsed)} test items")

    # build global triples from union of train+test parsed items
    print("Building global triples from inferential chains...")
    all_samples = train_parsed + test_parsed
    global_triples = build_global_triples(all_samples)
    print(f"Global triples count (deduplicated): {len(global_triples)}")

    # build calibration & test lists (UaG expects calibration_data entries)
    calib_items = build_calibration_items(train_parsed)
    test_items = build_test_items(test_parsed)

    if len(calib_items) == 0:
        raise RuntimeError("No calibration items extracted from training data. Check TopicEntityName/Answers fields.")

    # define lambda grid (example coarse grid)
    lambda_grid = [
        (0.4, 0.3, 0.2),
        (0.3, 0.25, 0.2),
        (0.3, 0.2, 0.1),
        (0.2, 0.2, 0.1),
        (0.15, 0.1, 0.05),
    ]
    target_alpha = args.alpha
    max_hop = args.max_hop

    print("Running LTT-style lambda search...")
    chosen_lambda, lambda_stats = ltt_lambda_search(calib_items, global_triples, lambda_grid, target_alpha, max_hop)
    print("\nLambda search summary:")
    for s in lambda_stats:
        print(f" lambda={s['lambda']}, ecr={s['ecr']:.3f}, avg_set={s['avg_size']:.3f}, valid={s['valid']}")
    print(f"\nChosen lambda: {chosen_lambda}")

    # fit final core on full calibration (train)
    print("\nTraining final UaGCore on full calibration using chosen lambda...")
    final_core = UaGCore(calibration_data=calib_items, global_triples=global_triples,
                         path_alpha=chosen_lambda[0], ans_alpha=chosen_lambda[1], post_alpha=chosen_lambda[2],
                         max_hop=max_hop)

    # evaluate on test
    print("\nEvaluating on test set...")
    test_eval = evaluate_core_on_dataset_with_conf(final_core, test_items)

    print(f"\nTest ECR: {test_eval['ecr']:.3f}, "
          f"AvgSet: {test_eval['avg_size']:.3f}, "
          f"MeanConf: {test_eval['mean_conf']:.3f}")

    print("\nPer-question sample results (first 10):")
    for d in test_eval["details"][:10]:
        print("-" * 80)
        print(f"QID: {d['id']}")
        print(f"Q: {d['question']}")
        print(f"Pred: {d['pred']}")
        print(f"True: {d['true']}")
        print(f"Covered: {d['covered']}")
        print(f"Average Confidence: {d['avg_conf']:.3f}")
        print("Answer Confidences:")
        for a, c in d["per_answer_conf"].items():
            print(f"   {a}: {c:.3f}")


    # optional: save results to JSON
    if args.out_json:
        # 将 set 类型转换为 list 类型以便 JSON 序列化
        serializable_test_eval = test_eval.copy()
        serializable_details = []

        for detail in serializable_test_eval["details"]:
            detail_copy = detail.copy()
            # 将 set 转换为 list
            if isinstance(detail_copy.get("pred"), set):
                detail_copy["pred"] = list(detail_copy["pred"])
            if isinstance(detail_copy.get("true"), set):
                detail_copy["true"] = list(detail_copy["true"])
            serializable_details.append(detail_copy)

        serializable_test_eval["details"] = serializable_details

        out = {
            "chosen_lambda": chosen_lambda,
            "lambda_stats": lambda_stats,
            "test_eval": serializable_test_eval
        }
        with open(args.out_json, "w", encoding="utf-8") as fw:
            json.dump(out, fw, ensure_ascii=False, indent=2)
        print(f"\nSaved results to {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="./data/WebQSP_train.json", help="path to WebQSP_train.json")
    parser.add_argument("--test", type=str, default="./data/WebQSP_test.json", help="path to WebQSP_test.json")
    parser.add_argument("--max_train", type=int, default=None, help="max train samples to use (for speed)")
    parser.add_argument("--max_test", type=int, default=None, help="max test samples to use (for speed)")
    parser.add_argument("--alpha", type=float, default=0.2, help="overall allowed error (target alpha)")
    parser.add_argument("--max_hop", type=int, default=2, help="maximum number of hops to explore")
    parser.add_argument("--out_json", type=str, default=None, help="optional output JSON file for results")
    args = parser.parse_args()
    main(args)
