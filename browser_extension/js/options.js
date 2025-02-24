// Default configuration
const DEFAULT_CONFIG = {
  apiUrl: 'http://localhost:8000/api',
  maxResults: 3
};

// Save settings
async function saveSettings(e) {
  e.preventDefault();
  
  const apiUrl = document.getElementById('apiUrl').value.trim();
  const maxResults = parseInt(document.getElementById('maxResults').value);
  
  // Validate URL
  try {
    new URL(apiUrl);
  } catch (e) {
    showStatus('Please enter a valid URL', false);
    return;
  }
  
  // Validate maxResults
  if (maxResults < 1 || maxResults > 10) {
    showStatus('Maximum results must be between 1 and 10', false);
    return;
  }
  
  // Save to chrome.storage
  try {
    await chrome.storage.sync.set({
      ragConfig: {
        apiUrl: apiUrl,
        maxResults: maxResults
      }
    });
    
    showStatus('Settings saved successfully!', true);
    
    // Test connection
    testConnection(apiUrl);
    
  } catch (error) {
    showStatus('Error saving settings: ' + error.message, false);
  }
}

// Load settings
async function loadSettings() {
  try {
    const result = await chrome.storage.sync.get('ragConfig');
    const config = result.ragConfig || DEFAULT_CONFIG;
    
    document.getElementById('apiUrl').value = config.apiUrl;
    document.getElementById('maxResults').value = config.maxResults;
    
    // Test connection on load
    testConnection(config.apiUrl);
    
  } catch (error) {
    showStatus('Error loading settings: ' + error.message, false);
  }
}

// Test API connection
async function testConnection(apiUrl) {
  try {
    const response = await fetch(`${apiUrl}/health`);
    if (response.ok) {
      showStatus('✓ Connected to RAG system', true);
    } else {
      showStatus('⚠️ Could not connect to RAG system', false);
    }
  } catch (error) {
    showStatus('⚠️ Could not connect to RAG system: ' + error.message, false);
  }
}

// Show status message
function showStatus(message, success) {
  const status = document.getElementById('status');
  status.textContent = message;
  status.className = 'status ' + (success ? 'success' : 'error');
  status.style.display = 'block';
  
  // Hide after 3 seconds
  setTimeout(() => {
    status.style.display = 'none';
  }, 3000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  document.getElementById('settingsForm').addEventListener('submit', saveSettings);
}); 