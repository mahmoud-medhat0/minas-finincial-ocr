import './style.css'

const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const templateSelector = document.getElementById('template-selector') as HTMLSelectElement;
const resultsGrid = document.getElementById('results') as HTMLDivElement;
const loading = document.getElementById('loading') as HTMLDivElement;
const rawTextSection = document.getElementById('raw-text-section') as HTMLElement;
const rawTextContent = document.getElementById('raw-text-content') as HTMLDivElement;
const previewSection = document.getElementById('preview-section') as HTMLDivElement;
const imagePreviewContainer = document.getElementById('image-preview-container') as HTMLDivElement;
const reprocessBtn = document.getElementById('reprocess-btn') as HTMLButtonElement;

const BASE_URL = 'http://localhost:8000';
const API_URL = `${BASE_URL}/process`;
const TEMPLATES_URL = `${BASE_URL}/templates`;

let currentFile: File | null = null;

// Fetch available templates
async function loadTemplates() {
  try {
    const response = await fetch(TEMPLATES_URL);
    const templates = await response.json();
    templates.forEach((name: string) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      templateSelector.appendChild(option);
    });
  } catch (error) {
    console.error('Error loading templates:', error);
  }
}

loadTemplates();

// Clipboard Support
window.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (items) {
    for (const item of items) {
      if (item.type.indexOf('image') !== -1) {
        const file = item.getAsFile();
        if (file) {
          handleFiles(createFileList(file));
        }
      }
    }
  }
});

// Helper to wrap single file in FileList-like object
function createFileList(file: File): FileList {
  const dt = new DataTransfer();
  dt.items.add(file);
  return dt.files;
}

// Reprocess Logic
reprocessBtn.addEventListener('click', () => {

  if (currentFile) {
    handleFiles(null, true);
  }
});

// Drag & Drop Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer?.files) {
    handleFiles(e.dataTransfer.files);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files) {
    handleFiles(fileInput.files);
  }
});

function renderPreview(file: File) {
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreviewContainer.innerHTML = `<img src="${e.target?.result}" alt="Preview">`;
    previewSection.setAttribute('data-hidden', 'false');
  };
  reader.readAsDataURL(file);
}

async function handleFiles(files: FileList | null, isReprocess = false) {
  if (!isReprocess && files) {
    currentFile = files[0];
  }

  if (!currentFile) return;

  if (!isReprocess) {
    renderPreview(currentFile);
  }

  // UI Reset
  resultsGrid.innerHTML = '';
  resultsGrid.classList.remove('visible');
  loading.style.display = 'block';
  rawTextSection.setAttribute('data-hidden', 'true');

  const formData = new FormData();
  formData.append('file', currentFile);


  try {
    const url = new URL(API_URL);
    if (templateSelector.value !== 'Auto-Detect') {
      url.searchParams.append('template', templateSelector.value);
    }

    const response = await fetch(url.toString(), {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Extraction failed');

    const data = await response.json();
    displayResults(data);
  } catch (error) {
    console.error(error);
    alert('Error processing file. Ensure the backend is running at :8000');
  } finally {
    loading.style.display = 'none';
  }
}

function displayResults(data: any) {
  const fields = [
    { label: 'Template', value: data.template, accent: true },
    { label: 'Amount', value: data.amount ? `${data.amount} ${data.currency || ''}` : 'Not Found' },
    { label: 'Transaction ID', value: data.transaction_id || 'Not Found' },
    { label: 'Date', value: data.date || 'Not Found' },
    { label: 'Sender', value: data.sender || 'Not Found' },
    { label: 'Sender Name', value: data.sender_name || 'Not Found' },
    { label: 'Receiver Account', value: data.receiver || 'Not Found' },
    { label: 'Receiver Name', value: data.receiver_name || 'Not Found' },
    { label: 'Status', value: data.status || 'Not Found' },
    { label: 'Type', value: data.transaction_type || 'Not Found' },
    { label: 'Comment', value: data.comment || 'Not Found' }
  ];

  resultsGrid.innerHTML = fields.map(f => `
        <div class="data-card">
            <h3>${f.label}</h3>
            <div class="value">${f.value}</div>
        </div>
    `).join('');

  // Confidence badge
  const confidenceHtml = `
        <div class="data-card">
            <h3>Confidence Score</h3>
            <span class="status-badge status-${data.confidence.toLowerCase()}">${data.confidence}</span>
        </div>
    `;
  resultsGrid.innerHTML += confidenceHtml;

  resultsGrid.classList.add('visible');

  // Raw text
  rawTextContent.textContent = data.raw_text;
  rawTextSection.removeAttribute('data-hidden');
}
