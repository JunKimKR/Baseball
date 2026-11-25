import { useEffect, useState } from "react";
import UploadPage from "./components/UploadPage";
import AnalysisPage from "./components/AnalysisPage";

/** Hash routes: `#/` → upload + history, `#/a/{id}` → analysis. */
export function parseRoute(hash: string): { page: "upload" } | { page: "analysis"; id: string } {
  const m = /^#\/a\/([^/?]+)/.exec(hash);
  return m ? { page: "analysis", id: decodeURIComponent(m[1]) } : { page: "upload" };
}

export default function App() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const on = () => setHash(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const route = parseRoute(hash);
  return (
    <main className="app">
      {route.page === "analysis" ? <AnalysisPage id={route.id} /> : <UploadPage />}
    </main>
  );
}
