// Re-render equations after Material's page navigation.
window.MathJax = {
  tex: {inlineMath: [["\\(", "\\)"], ["$", "$"]], displayMath: [["\\[", "\\]"], ["$$", "$$"]], processEscapes: true},
  options: {ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex"}
};
