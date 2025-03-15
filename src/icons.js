// Use a simpler approach with CDN instead of ES modules
document.addEventListener('DOMContentLoaded', () => {
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
    console.log('Lucide icons initialized successfully');
  } else {
    console.error('Lucide library not loaded. Make sure the CDN script is included properly.');
  }
});

// Function to refresh icons when content is dynamically added
export function refreshIcons() {
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
}
