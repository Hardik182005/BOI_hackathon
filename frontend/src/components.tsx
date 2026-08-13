import { useEffect, useState } from "react";
import { ApiError } from "./api";

export function usePoll<T>(fn: () => Promise<T>, deps: any[] = [], intervalMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setInterval> | undefined;
    const run = () =>
      fn()
        .then((d) => live && (setData(d), setError(null)))
        .catch((e: unknown) =>
          live && setError(e instanceof ApiError ? e.message : "unexpected error")
        )
        .finally(() => live && setLoading(false));
    run();
    if (intervalMs > 0) timer = setInterval(run, intervalMs);
    return () => {
      live = false;
      if (timer) clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading };
}

export function Loading({ what = "data" }: { what?: string }) {
  return <div className="loading">Loading {what}…</div>;
}

export function ErrorState({ msg }: { msg: string }) {
  return (
    <div className="error-state">
      <b>Could not load data.</b>
      <div style={{ marginTop: 6 }}>{msg}</div>
      <div style={{ marginTop: 6, fontSize: 12 }}>
        No placeholder numbers are shown - this dashboard only displays real
        pipeline artifacts.
      </div>
    </div>
  );
}

export function Empty({ msg }: { msg: string }) {
  return <div className="empty">{msg}</div>;
}

export function TierBadge({ tier }: { tier: string }) {
  return <span className={`tier ${tier}`}>{tier.replace("_", " ")}</span>;
}

/** Shown when the stored locked-test artifact belongs to a model that no longer
 *  serves. The headline PR-AUC is re-attributed by `lockedHeadline`, but the
 *  tier, calibration and budget panels on these pages still come from the
 *  stored file, and a reader has no way to tell that from the numbers alone.
 *  Saying it once, in the same words on every page, is cheaper than a footnote
 *  per panel and harder to miss than none. */
export function RetiredArtifactNotice(
  { head }: { head: { fromDeployedModel: boolean; model?: string; prAuc?: number } },
) {
  if (head.fromDeployedModel) return null;
  return (
    <div className="notice human">
      <b>Read this before the numbers:</b> the stored locked-test artifact was
      produced by a model the leakage firewall has since retired. The PR-AUC
      above is the <b>deployed</b> scorer{head.model ? <> (<code>{head.model}</code>)</> : null}
      {" "}on the same split. Panels below that read the artifact directly —
      risk-tier distribution, calibration, budget recall — still describe the
      retired run and are not the deployed model's behaviour.
    </div>
  );
}

export function HumanReviewNotice() {
  return (
    <div className="notice human">
      <b>Human-in-the-loop:</b> every review tier is a recommendation to a
      human analyst. MuleGuard never freezes accounts automatically, and a
      behavioural risk score is not proof of criminal intent.
    </div>
  );
}
