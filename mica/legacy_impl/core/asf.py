from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import json

class ASF:
    def __init__(self,
                 w_init: Dict[str, float] | None = None,
                 beta: float = 0.9,
                 persist: Optional[Path] = None,
                 K: int = 4,
                 steps: Optional[List[str]] = None,
                 leak: Optional[Dict[str, float]] = None,
                 clamp: Optional[Dict[str, float]] = None,
                 reward: float = 1.05,
                 penalty: float = 0.90,
                 conf_freeze: float = 0.80,
                 gamma: float = 2.0,
                 eta: float = 0.10,
                 trust: float = 0.15,
                 # prior terms
                 transitions: Optional[Dict[str, List[str]]] = None,
                 lambda_trans: float = 2.0,
                 lambda_rule: float = 1.0,
                 # anti-collapse
                 expo_rho: float = 0.5,
                 b_cap: float = 0.5,
                 floor_per_class: float = 0.02,
                 balance_window: int = 50,
                 balance_tau: float = 0.6):
        self.persist = Path(persist) if persist else None
        self.beta = float(beta)

        self.K = int(K)
        self.steps = steps[:] if steps else [f"S{i+1}" for i in range(self.K)]
        self.step2idx = {s.upper(): i for i, s in enumerate(self.steps)}

        wi = w_init or {"s": 0.5, "r": 0.5}
        gs = float(wi.get("s", 0.5)); gr = float(wi.get("r", 0.5))
        s = gs + gr
        self.g = [gs/s if s>0 else 0.5, gr/s if s>0 else 0.5]

        self.W = [[1.0/self.K, 1.0/self.K] for _ in range(self.K)]
        self.b = [0.0 for _ in range(self.K)]

        lk = leak or {"s": 0.05, "r": 0.05}
        self.leak_s = float(lk.get("s", 0.05))
        self.leak_r = float(lk.get("r", 0.05))
        cl = clamp or {"lo": 0.05, "hi": 0.95}
        self.lo = float(cl.get("lo", 0.05))
        self.hi = float(cl.get("hi", 0.95))
        self.reward = float(reward)
        self.penalty = float(penalty)

        self.conf_freeze = float(conf_freeze)
        self.gamma       = float(gamma)
        self.eta         = float(eta)
        self.trust       = float(trust)

        self.transitions  = transitions or {}
        self.lambda_trans = float(lambda_trans)
        self.lambda_rule  = float(lambda_rule)

        self.n_seen   = [0 for _ in range(self.K)]
        self.expo_rho = float(expo_rho)
        self.b_cap    = float(b_cap)
        self.floor    = float(floor_per_class)
        self.hist_win = int(balance_window)
        self.hist_tau = float(balance_tau)
        self.recent_sf: List[str] = []

        self._load()

    # IO
    def _load(self):
        if not self.persist or not self.persist.exists(): return
        try:
            data = json.loads(self.persist.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "s" in data and "r" in data:
                gs, gr = float(data["s"]), float(data["r"]); s = gs+gr
                self.g = [gs/s if s>0 else 0.5, gr/s if s>0 else 0.5]
                print(f"[ASF] loaded legacy g from {self.persist}: {self.g}"); return
            if all(k in data for k in ("K","steps","W","b","g")):
                self.K = int(data["K"]); self.steps = list(data["steps"])
                self.step2idx = {s.upper(): i for i, s in enumerate(self.steps)}
                self.W = [[float(x) for x in row] for row in data["W"]]
                self.b = [float(x) for x in data["b"]]
                g = [float(x) for x in data["g"]]; s = g[0]+g[1]
                self.g = [g[0]/s if s>0 else 0.5, g[1]/s if s>0 else 0.5]
                print(f"[ASF] loaded CLLF params from {self.persist}")
        except Exception as e:
            print(f"[ASF] load failed: {e}")

    def _save(self):
        if not self.persist: return
        try:
            obj = {"version": 2, "K": self.K, "steps": self.steps,
                   "W": self.W, "b": self.b, "g": self.g}
            self.persist.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        except Exception:
            pass

    # helpers
    def _idx(self, step: str) -> Optional[int]:
        return self.step2idx.get(str(step).strip().upper(), None)

    def _normalize(self):
        # per-column clamp + normalize
        for e in (0,1):
            col = [max(self.lo, min(self.hi, float(self.W[j][e]))) for j in range(self.K)]
            s = sum(col)
            for j in range(self.K):
                self.W[j][e] = col[j]/s if s>0 else 1.0/self.K
        # floor and re-norm
        for e in (0,1):
            col = [max(self.floor, self.W[j][e]) for j in range(self.K)]
            s = sum(col)
            for j in range(self.K):
                self.W[j][e] = col[j]/s if s>0 else 1.0/self.K
        # bias clip
        for j in range(self.K):
            self.b[j] = max(-self.b_cap, min(self.b_cap, self.b[j]))
        # gates
        gs = max(self.lo, min(self.hi, self.g[0])); gr = max(self.lo, min(self.hi, self.g[1]))
        s = gs+gr; self.g = [gs/s, gr/s] if s>0 else [0.5, 0.5]

    def _impact(self, C: float, hit: bool) -> float:
        C = max(0.0, min(1.0, float(C)))
        if hit and (C >= self.conf_freeze): return 0.0
        return pow(1.0 - C, self.gamma)

    def _mul_trust(self, x: float, delta: float) -> float:
        lo, hi = 1.0 - self.trust, 1.0 + self.trust
        f = 1.0 + delta
        if f < lo: f = lo
        if f > hi: f = hi
        return x * f

    # scoring
    def _scores(self, s_pred, s_conf, r_pred, r_conf,
                compat: Optional[List[float]] = None,
                prev_step: Optional[str] = None) -> List[float]:
        si = self._idx(s_pred); ri = self._idx(r_pred)
        Cs = float(s_conf or 0.0); Cr = float(r_conf or 0.0)
        allow = None
        if prev_step and self.transitions:
            allow = set([a.upper() for a in self.transitions.get(str(prev_step).strip().upper(), [])])
        scores = []
        for j in range(self.K):
            csj = Cs if (si == j) else self.leak_s * Cs
            crj = Cr if (ri == j) else self.leak_r * Cr
            val = self.b[j] + self.g[0]*self.W[j][0]*csj + self.g[1]*self.W[j][1]*crj
            if compat is not None and j < len(compat):
                val += self.lambda_rule * float(compat[j])
            if allow is not None:
                st = self.steps[j].upper()
                if st not in allow:
                    val -= self.lambda_trans
            scores.append(val)
        return scores

    def fuse(self, s_pred: Tuple[str, float], r_pred: Tuple[str, float],
             prev_step: Optional[str] = None, compat: Optional[List[float]] = None)\
             -> Tuple[str, float, Dict[str, Any]]:
        (S_s, C_s) = s_pred; (S_r, C_r) = r_pred
        scores = self._scores(S_s, C_s, S_r, C_r, compat=compat, prev_step=prev_step)
        j = max(range(self.K), key=lambda k: scores[k]) if scores else 0
        S_f = self.steps[j]
        exps = [pow(2.718281828, x - scores[j]) for x in scores]; denom = sum(exps) or 1.0
        conf = 1.0/denom
        # maintain recent histogram
        self.recent_sf.append(S_f)
        if len(self.recent_sf) > self.hist_win: self.recent_sf.pop(0)
        meta = {"scores": {self.steps[i]: scores[i] for i in range(self.K)},
                "g_s": self.g[0], "g_r": self.g[1], "Cs": C_s, "Cr": C_r}
        return S_f, float(conf), meta

    # legacy
    def update_with_feedback(self, chosen: str, user_step: str, predicted_step: str):
        if str(predicted_step).strip().upper() != str(user_step).strip().upper():
            if (chosen or "s") == "s": self.g[0] *= self.beta
            else: self.g[1] *= self.beta
            self._normalize(); self._save()

    # confidence-aware + anti-collapse
    def update_with_feedback_plus(self, user_step, s_pred, r_pred, chosen, fused,
                                  s_conf: float = 0.0, r_conf: float = 0.0):
        y  = str(user_step).strip().upper()
        yi = self._idx(y); si = self._idx(s_pred); ri = self._idx(r_pred)
        if yi is None: return
        # exposure decay + balance inhibition
        self.n_seen[yi] += 1
        expo = 1.0 / pow(max(1, self.n_seen[yi]), self.expo_rho)
        damp = 1.0
        if self.recent_sf:
            cnt = sum(1 for s in self.recent_sf if s == y)
            if cnt / len(self.recent_sf) > self.hist_tau:
                damp = 0.5
        imp_s_hit = self._impact(s_conf, hit=(si == yi))
        imp_r_hit = self._impact(r_conf, hit=(ri == yi))
        imp_s_err = self._impact(s_conf, hit=False) if si != yi else 0.0
        imp_r_err = self._impact(r_conf, hit=False) if ri != yi else 0.0
        eta = self.eta * expo * damp

        if (si == yi) and (ri != yi):
            self.W[yi][0] = self._mul_trust(self.W[yi][0],  + eta * imp_s_hit)
            if ri is not None and ri != yi:
                self.W[ri][1] = self._mul_trust(self.W[ri][1], - eta * imp_r_err)
            self.g[0] = self._mul_trust(self.g[0], + 0.02 * imp_s_hit)
            self.g[1] = self._mul_trust(self.g[1], - 0.02 * imp_r_err)
        elif (ri == yi) and (si != yi):
            self.W[yi][1] = self._mul_trust(self.W[yi][1],  + eta * imp_r_hit)
            if si is not None and si != yi:
                self.W[si][0] = self._mul_trust(self.W[si][0], - eta * imp_s_err)
            self.g[1] = self._mul_trust(self.g[1], + 0.02 * imp_r_hit)
            self.g[0] = self._mul_trust(self.g[0], - 0.02 * imp_s_err)
        elif (ri == yi) and (si == yi):
            self.W[yi][0] = self._mul_trust(self.W[yi][0], + eta * imp_s_hit)
            self.W[yi][1] = self._mul_trust(self.W[yi][1], + eta * imp_r_hit)
        else:
            margin = float(s_conf) - float(r_conf)
            if margin >= 0.01 and ri is not None:
                self.W[yi][1] = self._mul_trust(self.W[yi][1], + eta * imp_r_err)
                if ri != yi:
                    self.W[ri][1] = self._mul_trust(self.W[ri][1], - eta * imp_r_err)
            elif margin <= -0.01 and si is not None:
                self.W[yi][0] = self._mul_trust(self.W[yi][0], + eta * imp_s_err)
                if si != yi:
                    self.W[si][0] = self._mul_trust(self.W[si][0], - eta * imp_s_err)
            else:
                self.W[yi][0] = self._mul_trust(self.W[yi][0], + eta * imp_s_err * self.g[0])
                self.W[yi][1] = self._mul_trust(self.W[yi][1], + eta * imp_r_err * self.g[1])

        # bias with cap
        alpha = 0.01 * expo * damp * max(imp_s_hit, imp_r_hit, imp_s_err, imp_r_err)
        self.b[yi] = max(-self.b_cap, min(self.b_cap, self.b[yi] + alpha))
        minus = alpha / max(1, self.K-1)
        for j in range(self.K):
            if j != yi:
                self.b[j] = max(-self.b_cap, min(self.b_cap, self.b[j] - minus))

        self._normalize()
        self._save()

    @property
    def w(self):
        return {"s": self.g[0], "r": self.g[1]}
