async function caricaSnippet(el) {
  const slug = el.dataset.snippet;
  if (!slug) return;
  try {
    const highlighted = await fetch(`assets/highlighted_snippets/${slug}.html`);
    if (highlighted.ok) {
      const container = document.createElement("div");
      container.className = "highlighted-code";
      container.innerHTML = await highlighted.text();
      const pre = el.closest("pre");
      if (pre) {
        pre.replaceWith(container);
      } else {
        el.replaceWith(container);
      }
      return;
    }
    const response = await fetch(`assets/snippets/${slug}.txt`);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    el.textContent = await response.text();
  } catch (error) {
    el.textContent =
      `Impossibile caricare assets/snippets/${slug}.txt\n\n` +
      `Servire il report con un server locale, ad esempio:\n` +
      `python3 -m http.server 8000\n\n` +
      `Errore: ${error.message}`;
  }
}

async function caricaHighlightedSnippet(el) {
  const slug = el.dataset.highlightedSnippet;
  if (!slug) return;
  try {
    const response = await fetch(`assets/highlighted_snippets/${slug}.html`);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    el.innerHTML = await response.text();
  } catch (error) {
    el.textContent =
      `Impossibile caricare assets/highlighted_snippets/${slug}.html\n\n` +
      `Rigenerare con: python highlight_snippets.py\n\n` +
      `Errore: ${error.message}`;
  }
}

function setupToggle() {
  document.querySelectorAll("[data-toggle-code]").forEach((button) => {
    button.addEventListener("click", () => {
      const body = button.closest(".code-card").querySelector(".highlighted-code, pre");
      if (!body) return;
      const expanded = body.style.maxHeight === "none";
      body.style.maxHeight = expanded ? "590px" : "none";
      button.textContent = expanded ? "espandi" : "comprimi";
    });
  });
}

function setupCopy() {
  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.querySelector(button.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent.trim());
        const label = button.textContent;
        button.textContent = "copiato";
        setTimeout(() => { button.textContent = label; }, 1200);
      } catch (_error) {
        button.textContent = "copy failed";
      }
    });
  });
}

function setupIndiceAttivo() {
  const links = [...document.querySelectorAll(".toc a[href^='#']")];
  const byId = new Map(links.map((link) => [link.getAttribute("href").slice(1), link]));
  const sections = [...document.querySelectorAll("main section[id]")];
  if (!("IntersectionObserver" in window) || sections.length === 0) return;

  const observer = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach((link) => link.classList.remove("active"));
    const active = byId.get(visible.target.id);
    if (active) active.classList.add("active");
  }, { rootMargin: "-18% 0px -70% 0px", threshold: [0.05, 0.2, 0.5] });

  sections.forEach((section) => observer.observe(section));
}

document.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([
    ...[...document.querySelectorAll("[data-highlighted-snippet]")].map(caricaHighlightedSnippet),
    ...[...document.querySelectorAll("code[data-snippet]")].map(caricaSnippet),
  ]);
  setupToggle();
  setupCopy();
  setupIndiceAttivo();
});
