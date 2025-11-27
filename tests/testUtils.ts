
const container = document.getElementById("test-container");

export async function addTest(name: string, fn: (log: (msg: string) => void) => Promise<[boolean, string]>) {
  if (!container) return;

  const details = document.createElement("details");
  details.open = false;
  details.style.marginBottom = "10px";
  details.style.border = "1px solid #ccc";
  details.style.borderRadius = "4px";
  details.style.padding = "10px";

  const summary = document.createElement("summary");
  summary.style.cursor = "pointer";
  summary.style.fontWeight = "bold";
  summary.textContent = `⏳ ${name}`;

  const content = document.createElement("div");
  content.style.marginTop = "10px";
  content.style.fontFamily = "monospace";
  content.style.whiteSpace = "pre-wrap";
  content.textContent = "Running...";

  const logContainer = document.createElement("div");
  logContainer.style.marginTop = "10px";
  logContainer.style.fontFamily = "monospace";
  logContainer.style.fontSize = "0.9em";
  logContainer.style.color = "#555";
  logContainer.style.whiteSpace = "pre-wrap";
  logContainer.style.borderTop = "1px solid #eee";
  logContainer.style.paddingTop = "5px";
  logContainer.style.display = "none";

  details.appendChild(summary);
  details.appendChild(content);
  details.appendChild(logContainer);
  container.appendChild(details);

  const log = (msg: string) => {
      logContainer.style.display = "block";
      logContainer.textContent += msg + "\n";
  };

  try {
    const [success, message] = await fn(log);
    if (success) {
      summary.textContent = `✅ ${name}`;
      summary.style.color = "green";
      details.style.borderColor = "green";
      details.style.backgroundColor = "#f0fff4";
      content.textContent = message || "Passed";
    } else {
      summary.textContent = `❌ ${name}`;
      summary.style.color = "red";
      details.style.borderColor = "red";
      details.style.backgroundColor = "#fff5f5";
      content.textContent = message || "Failed";
    }
  } catch (e: any) {
    summary.textContent = `❌ ${name}`;
    summary.style.color = "red";
    details.style.borderColor = "red";
    details.style.backgroundColor = "#fff5f5";
    content.textContent = `Exception: ${e.message}\n${e.stack}`;
    log(`Exception: ${e.message}\n${e.stack}`);
  }
}

export async function addInfo(
  name: string,
  fn: (log: (msg: string) => void) => Promise<void>
) {
  if (!container) return;

  const details = document.createElement("details");
  details.open = false;
  details.style.marginBottom = "10px";
  details.style.border = "1px solid #90cdf4"; // Light blue border
  details.style.borderRadius = "4px";
  details.style.padding = "10px";
  details.style.backgroundColor = "#ebf8ff"; // Light blue background

  const summary = document.createElement("summary");
  summary.style.cursor = "pointer";
  summary.style.fontWeight = "bold";
  summary.textContent = `ℹ️ ${name}`;
  summary.style.color = "#2b6cb0"; // Darker blue text

  const content = document.createElement("div");
  content.style.marginTop = "10px";
  content.style.fontFamily = "monospace";
  content.style.whiteSpace = "pre-wrap";
  content.textContent = "";

  details.appendChild(summary);
  details.appendChild(content);
  container.appendChild(details);

  const log = (msg: string) => {
    content.textContent += msg + "\n";
  };

  try {
    await fn(log);
  } catch (e: any) {
    log(`❌ Error: ${e.message}\n${e.stack}`);
    details.style.borderColor = "red";
    details.style.backgroundColor = "#fff5f5";
  }
}
