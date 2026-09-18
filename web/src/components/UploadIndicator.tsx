import { Link, useLocation } from "react-router-dom";
import { pct } from "../format";
import { overallProgress, useUploads } from "../uploadStore";

/** A small badge in the top bar while an upload runs or waits to be resumed. */
export default function UploadIndicator() {
  const uploads = useUploads();
  const { pathname } = useLocation();
  if (uploads.phase === "draft" || pathname === "/upload") return null;
  const progress = pct(overallProgress(uploads));
  return (
    <Link to="/upload" className="badge badge-busy" style={{ textDecoration: "none" }}>
      {uploads.phase === "uploading" ? `Uploading ${progress}` : `Upload paused at ${progress}`}
    </Link>
  );
}
