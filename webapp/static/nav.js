function setupBackButtons() {
  for (const btn of document.querySelectorAll("[data-back-fallback]")) {
    btn.addEventListener("click", () => {
      const fallback = btn.dataset.backFallback || "/";
      if (window.history.length > 1) {
        window.history.back();
      } else {
        window.location.href = fallback;
      }
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", setupBackButtons);
} else {
  setupBackButtons();
}
