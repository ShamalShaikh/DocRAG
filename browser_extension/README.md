# RAG System Browser Extension

A browser extension that allows you to query your local RAG (Retrieval-Augmented Generation) system while browsing.

## Features

- Query your RAG system directly from the browser
- View answers with source documents
- See a tag cloud of document topics
- Configure API endpoint for local or remote access
- Real-time connection status indicator

## Installation

### Chrome/Edge

1. Open Chrome/Edge and navigate to `chrome://extensions` or `edge://extensions`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select the `browser_extension` folder

### Firefox

1. Open Firefox and navigate to `about:debugging`
2. Click "This Firefox" in the sidebar
3. Click "Load Temporary Add-on"
4. Select any file in the `browser_extension` folder

## Configuration

1. Click the extension icon in your browser toolbar
2. Click the ⚙️ (settings) button
3. Configure the following settings:
   - API URL: Default is `http://localhost:8000/api` for local development
   - Maximum Results: Number of source documents to display (1-10)
4. Click "Save Settings"

## Usage

1. Click the extension icon to open the popup
2. Enter your question in the text area
3. Click "Search" or press Enter
4. View the answer and source documents
5. Check the tag cloud for popular topics in your knowledge base

## Development

### File Structure

```
browser_extension/
├── manifest.json        # Extension configuration
├── popup.html          # Main popup interface
├── options.html        # Settings page
├── css/
│   └── popup.css       # Styles for popup
├── js/
│   ├── popup.js        # Popup functionality
│   └── options.js      # Settings functionality
└── images/            # Extension icons
```

### Local Development

1. Ensure your RAG system's FastAPI server is running at `http://localhost:8000`
2. Make changes to the extension files
3. Reload the extension in your browser:
   - Chrome/Edge: Click the refresh icon on the extension card
   - Firefox: Click "Reload" next to the extension

### Using with a Remote API

To use the extension with a remote API:

1. Deploy your RAG system to a publicly accessible server
2. Open the extension settings
3. Update the API URL to your server's URL
4. Ensure CORS is properly configured on your server

## Troubleshooting

- **Connection Issues**: Check if your RAG system's FastAPI server is running and accessible
- **CORS Errors**: Ensure your server's CORS settings allow requests from the extension
- **Loading Failed**: Try reloading the extension or check browser console for errors

## Security Notes

- The extension only requires permissions for:
  - `storage`: To save settings
  - `activeTab`: To interact with the current tab
  - `http://localhost:8000/*`: To communicate with your local RAG system
- No data is collected or sent to third parties
- All communication is directly between the extension and your RAG system

## Page Scraping Feature

The extension now includes the ability to save the current page to your RAG system:

1. Click the extension icon while on any webpage
2. Click the "Save Current Page" button
3. The extension will:
   - Extract the main content from the page
   - Remove unnecessary elements (ads, navigation, etc.)
   - Capture metadata (title, URL, author if available)
   - Send the content to your RAG system for processing

### Error Handling

The page scraping feature includes several safeguards:
- Large pages are automatically truncated to avoid token limits
- Restricted content (iframes, etc.) is automatically skipped
- Clear error messages for common issues
- Automatic retry for temporary failures

### Troubleshooting Page Scraping

If you encounter issues:
1. Ensure the page is fully loaded before scraping
2. Check if the page allows content extraction
3. For very large pages, try selecting and saving specific sections
4. Check the browser console for detailed error messages 