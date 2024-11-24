document.addEventListener("DOMContentLoaded", function () {
  const tocHeaders = document.querySelectorAll(".toc-h2");

  tocHeaders.forEach((header) => {
    header.addEventListener("click", function () {
      // Toggle the 'open' class on the clicked .toc-h2
      this.classList.toggle("open");

      // Find the next <ul> sibling and toggle its 'visible' class
      const nextUl = this.nextElementSibling;
      if (nextUl && nextUl.tagName === "UL") {
        nextUl.classList.toggle("visible");
      }
    });
  });
});
