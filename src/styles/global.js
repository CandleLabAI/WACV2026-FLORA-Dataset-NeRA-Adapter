document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('goTop');

  if (!btn) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 300) {
      btn.classList.add('visible');
    } else {
      btn.classList.remove('visible');
    }
  });

  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});
document.addEventListener("DOMContentLoaded", () => {
  const menuButton = document.getElementById("menuButton");
  const menuDropdown = document.getElementById("menuDropdown");

  if (!menuButton || !menuDropdown) return;

  // Toggle dropdown
  menuButton.addEventListener("click", (e) => {
    e.stopPropagation(); // prevent document click from firing
    const isOpen = menuDropdown.classList.contains("opacity-100");

    menuDropdown.classList.toggle("opacity-100", !isOpen);
    menuDropdown.classList.toggle("visible", !isOpen);
    menuDropdown.classList.toggle("invisible", isOpen);

    const arrow = menuButton.querySelector("svg");
    if (arrow) arrow.classList.toggle("rotate-180", !isOpen);
  });

  // Close dropdown if clicked outside
  document.addEventListener("click", (e) => {
    if (!menuButton.contains(e.target) && !menuDropdown.contains(e.target)) {
      menuDropdown.classList.remove("opacity-100");
      menuDropdown.classList.add("invisible");

      const arrow = menuButton.querySelector("svg");
      if (arrow) arrow.classList.remove("rotate-180");
    }
  });
});
