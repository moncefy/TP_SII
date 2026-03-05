const form = document.getElementById("loginForm");
const statusMessage = document.getElementById("statusMessage");

function setStatus(message, kind = "") {
  statusMessage.className = "status-message";
  if (kind) {
    statusMessage.classList.add(kind);
  }
  statusMessage.textContent = message;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  setStatus("Veryfing login...", "info");

  try {
    const response = await fetch("/api/login-success", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      }
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.message || "Failed to send email.");
    }

    setStatus("Login successful.", "success");
    form.reset();
  } catch (error) {
    setStatus(error.message || "Could not send email. Check server setup.", "error");
  }
});
