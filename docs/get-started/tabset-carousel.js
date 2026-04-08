// Enhances any .panel-tabset.carousel-tabset with prev/next arrow navigation
// and a smooth sliding transition between panes.
//
// Usage: add class="carousel-tabset" to a Quarto panel-tabset div, then
// include this script at the bottom of the page:
//   <script src="tabset-carousel.js"></script>

(function () {
  document.querySelectorAll(".panel-tabset.carousel-tabset").forEach(tabset => {
    const tabContent = tabset.querySelector(".tab-content");
    const navLinks = Array.from(tabset.querySelectorAll(".nav-link"));
    const panes = Array.from(tabContent.querySelectorAll(".tab-pane"));

    let animating = false;
    let currentIndex = navLinks.findIndex(t => t.classList.contains("active"));
    if (currentIndex === -1) currentIndex = 0;

    // Build a flex slide track containing all panes side by side
    const track = document.createElement("div");
    track.className = "tabset-slide-track";
    panes.forEach(pane => track.appendChild(pane));
    tabContent.appendChild(track);

    function paneWidth() {
      return panes[0].offsetWidth;
    }

    // Position track at the initial active pane with no animation
    track.style.transition = "none";
    track.style.transform = `translateX(${-currentIndex * paneWidth()}px)`;

    // Keep position correct on window resize
    window.addEventListener("resize", () => {
      track.style.transition = "none";
      track.style.transform = `translateX(${-currentIndex * paneWidth()}px)`;
    });

    // Inject arrow buttons flanking the tab content
    const prev = document.createElement("button");
    prev.className = "tabset-arrow";
    prev.setAttribute("aria-label", "Previous tab");
    prev.innerHTML = "&#8592;";

    const next = document.createElement("button");
    next.className = "tabset-arrow";
    next.setAttribute("aria-label", "Next tab");
    next.innerHTML = "&#8594;";

    const wrapper = document.createElement("div");
    wrapper.className = "tabset-content-wrapper";
    tabContent.parentNode.insertBefore(wrapper, tabContent);
    wrapper.appendChild(prev);
    wrapper.appendChild(tabContent);
    wrapper.appendChild(next);

    function updateButtons() {
      prev.disabled = currentIndex <= 0;
      next.disabled = currentIndex >= navLinks.length - 1;
    }

    function switchTo(toIndex) {
      if (animating || toIndex === currentIndex) return;
      if (toIndex < 0 || toIndex >= panes.length) return;

      animating = true;
      currentIndex = toIndex;

      navLinks.forEach((t, i) => t.classList.toggle("active", i === toIndex));
      updateButtons();

      track.style.transition = "transform 0.35s cubic-bezier(0.4, 0, 0.2, 1)";
      track.style.transform = `translateX(${-toIndex * paneWidth()}px)`;

      track.addEventListener("transitionend", function handler() {
        track.removeEventListener("transitionend", handler);
        animating = false;
      });
    }

    // Intercept nav link clicks — stop both immediate and bubbled handlers
    // so Bootstrap's delegated tab handler doesn't interfere
    navLinks.forEach((link, i) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        switchTo(i);
      }, true);
    });

    prev.addEventListener("click", () => switchTo(currentIndex - 1));
    next.addEventListener("click", () => switchTo(currentIndex + 1));

    updateButtons();
  });
})();
