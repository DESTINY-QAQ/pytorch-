const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const form = document.querySelector("#predict-form");
const emptyState = document.querySelector("#empty-state");
const preview = document.querySelector("#client-preview");
const previewImage = document.querySelector("#preview-image");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const resetButton = document.querySelector("#reset-button");
const submitButton = document.querySelector("#submit-button");
const allowedTypes = ["image/jpeg", "image/png", "image/bmp", "image/webp"];

function readableSize(bytes) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function showFile(file) {
  if (!file || !allowedTypes.includes(file.type) || file.size > 10 * 1024 * 1024) {
    input.value = "";
    window.alert("请选择不超过 10 MB 的 JPG、PNG、BMP 或 WEBP 图片。");
    return;
  }
  const reader = new FileReader();
  reader.onload = event => {
    previewImage.src = event.target.result;
    fileName.textContent = file.name;
    fileSize.textContent = readableSize(file.size);
    emptyState.hidden = true;
    preview.hidden = false;
    resetButton.disabled = false;
    submitButton.disabled = false;
  };
  reader.readAsDataURL(file);
}

input.addEventListener("change", () => showFile(input.files[0]));

["dragenter", "dragover"].forEach(eventName => {
  dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  showFile(file);
});

resetButton.addEventListener("click", () => {
  input.value = "";
  previewImage.removeAttribute("src");
  preview.hidden = true;
  emptyState.hidden = false;
  resetButton.disabled = true;
  submitButton.disabled = true;
});

form.addEventListener("submit", () => {
  submitButton.classList.add("loading");
  submitButton.disabled = true;
  resetButton.disabled = true;
});
