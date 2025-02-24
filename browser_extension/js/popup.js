// Default configuration
const DEFAULT_CONFIG = {
  apiUrl: 'http://localhost:8000/api',
  maxResults: 3
};

// Get stored configuration or use defaults
async function getConfig() {
  const result = await chrome.storage.sync.get('ragConfig');
  return result.ragConfig || DEFAULT_CONFIG;
}

// Update status indicator
function updateStatus(isOnline) {
  const indicator = document.getElementById('statusIndicator');
  indicator.style.color = isOnline ? '#28a745' : '#dc3545';
  indicator.title = isOnline ? 'Connected to RAG System' : 'Not Connected';
}

// Check API health
async function checkHealth() {
  try {
    const config = await getConfig();
    const response = await fetch(`${config.apiUrl}/health`);
    const isHealthy = response.ok;
    updateStatus(isHealthy);
    return isHealthy;
  } catch (error) {
    console.error('Health check failed:', error);
    updateStatus(false);
    return false;
  }
}

// Format sources for display
function formatSources(sources) {
  return sources.map((source, index) => `
    <div class="source-item">
      <strong>Source ${index + 1}</strong> (Score: ${source.similarity_score.toFixed(3)})<br>
      ${source.metadata.title || 'Untitled'}<br>
      <small>${source.content.substring(0, 150)}...</small>
    </div>
  `).join('');
}

// Handle search
async function handleSearch() {
  const queryInput = document.getElementById('queryInput');
  const answerDiv = document.getElementById('answer');
  const sourcesDiv = document.getElementById('sources');
  const spinner = document.getElementById('loadingSpinner');
  const query = queryInput.value.trim();

  if (!query) {
    answerDiv.innerHTML = '<em>Please enter a question.</em>';
    return;
  }

  try {
    spinner.style.display = 'block';
    answerDiv.innerHTML = '';
    sourcesDiv.innerHTML = '';

    const config = await getConfig();
    const response = await fetch(`${config.apiUrl}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        max_results: config.maxResults
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const data = await response.json();
    
    // Display answer
    answerDiv.innerHTML = `<p>${data.answer}</p>`;
    
    // Display sources
    if (data.sources && data.sources.length > 0) {
      sourcesDiv.innerHTML = `
        <h3>Sources:</h3>
        ${formatSources(data.sources)}
      `;
    }

    // Update visualization if available
    updateVisualization();

  } catch (error) {
    console.error('Search failed:', error);
    answerDiv.innerHTML = `<em>Error: ${error.message}</em>`;
  } finally {
    spinner.style.display = 'none';
  }
}

// Update visualization
async function updateVisualization() {
  const vizDiv = document.getElementById('visualization');
  try {
    const config = await getConfig();
    const response = await fetch(`${config.apiUrl}/embeddings`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch visualization data');
    }

    const data = await response.json();
    
    // If we have embeddings data, create a simple tag cloud
    if (data.metadata && data.metadata.length > 0) {
      const tags = new Map();
      
      // Count tag occurrences
      data.metadata.forEach(meta => {
        if (meta.tags) {
          meta.tags.forEach(tag => {
            tags.set(tag, (tags.get(tag) || 0) + 1);
          });
        }
      });

      // Create tag cloud HTML
      const tagCloudHtml = Array.from(tags.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([tag, count]) => `
          <span class="tag" style="font-size: ${Math.max(12, Math.min(20, 12 + count * 2))}px">
            ${tag} (${count})
          </span>
        `)
        .join(' ');

      vizDiv.innerHTML = `
        <h3>Top Topics</h3>
        <div class="tag-cloud">${tagCloudHtml}</div>
      `;
    }
  } catch (error) {
    console.error('Visualization update failed:', error);
    vizDiv.innerHTML = '<em>Visualization not available</em>';
  }
}

// Add scraping button handler
async function handlePageScrape() {
    const scrapingStatus = document.getElementById('scrapingStatus');
    const scrapeButton = document.getElementById('scrapePage');
    
    try {
        // Show loading state
        scrapingStatus.style.display = 'block';
        scrapeButton.disabled = true;
        
        // Get current tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        if (!tab) {
            throw new Error('No active tab found');
        }

        // Ensure content script is injected
        try {
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ['js/content.js']
            });
        } catch (error) {
            console.log('Content script already loaded or injection failed:', error);
            // Continue execution as the script might already be loaded
        }

        // Add timeout for message response
        const response = await Promise.race([
            chrome.tabs.sendMessage(tab.id, { action: "scrape_page" }),
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Content script response timeout')), 5000)
            )
        ]);
        
        if (!response || !response.success) {
            throw new Error(response?.error || 'Failed to extract page content');
        }
        
        // Get API configuration
        const config = await getConfig();
        
        // Send to API
        const apiResponse = await fetch(`${config.apiUrl}/scrape`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                url: response.content.metadata.url,
                metadata: response.content.metadata,
                html_content: response.content.html
            })
        });
        
        if (!apiResponse.ok) {
            const error = await apiResponse.json();
            throw new Error(error.detail || 'Failed to save page');
        }
        
        // Show success message
        scrapingStatus.innerHTML = `
            <p style="color: #28a745;">✓ Page saved successfully!</p>
        `;
        
    } catch (error) {
        console.error('Scraping failed:', error);
        scrapingStatus.innerHTML = `
            <p style="color: #dc3545;">Error: ${error.message}</p>
        `;
    } finally {
        // Reset after 3 seconds
        setTimeout(() => {
            scrapingStatus.style.display = 'none';
            scrapingStatus.innerHTML = `
                <div class="spinner"></div>
                <p>Processing page...</p>
            `;
            scrapeButton.disabled = false;
        }, 3000);
    }
}

// Initialize popup
document.addEventListener('DOMContentLoaded', async () => {
  // Add event listeners
  document.getElementById('searchButton').addEventListener('click', handleSearch);
  document.getElementById('queryInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  });
  
  document.getElementById('settingsButton').addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
  });

  // Add scraping button handler
  document.getElementById('scrapePage').addEventListener('click', handlePageScrape);

  // Check health and update status
  await checkHealth();
  
  // Initial visualization update
  updateVisualization();
}); 