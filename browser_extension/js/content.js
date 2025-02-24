// Content script for RAG Assistant
// This script runs in the context of web pages

/**
 * Extract the main content from the current page
 * @returns {Object} Object containing page content and metadata
 */
function extractPageContent() {
    // Get main content area if possible
    const mainContent = 
        document.querySelector('main') || 
        document.querySelector('article') || 
        document.querySelector('.content') ||
        document.body;

    // Remove unwanted elements to avoid token bloat
    const contentClone = mainContent.cloneNode(true);
    const unwantedSelectors = [
        'script', 'style', 'iframe', 'nav', 'footer',
        '[role="navigation"]', '[role="banner"]',
        '.ad', '.advertisement', '.social-share'
    ];
    
    unwantedSelectors.forEach(selector => {
        contentClone.querySelectorAll(selector).forEach(el => el.remove());
    });

    // Extract metadata
    const metadata = {
        title: document.title,
        url: window.location.href,
        timestamp: new Date().toISOString(),
        author: document.querySelector('meta[name="author"]')?.content || '',
        description: document.querySelector('meta[name="description"]')?.content || ''
    };

    return {
        html: contentClone.innerHTML,
        metadata: metadata
    };
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "scrape_page") {
        try {
            const content = extractPageContent();
            sendResponse({ success: true, content });
        } catch (error) {
            sendResponse({ 
                success: false, 
                error: "Failed to extract page content: " + error.message 
            });
        }
    }
    return true; // Required for async response
}); 