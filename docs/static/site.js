'use strict';
const copyButton = document.getElementById('copy-citation');
copyButton.addEventListener('click', async () => {
  const text = document.getElementById('citation').textContent;
  const status = document.getElementById('copy-status');
  try {
    if (!navigator.clipboard) throw new Error('Clipboard unavailable');
    await navigator.clipboard.writeText(text);
    status.textContent = 'Citation copied.';
  } catch (_) {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('citation'));
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
const dialog = document.getElementById('image-dialog');
const expanded = document.getElementById('expanded-image');
document.querySelectorAll('.figure-zoom').forEach(link => {
  link.addEventListener('click', event => {
    if (!dialog.showModal) return;
    event.preventDefault();
    expanded.src = link.href;
    expanded.alt = link.querySelector('img').alt;
    dialog.showModal();
  });
});
document.getElementById('close-image').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
