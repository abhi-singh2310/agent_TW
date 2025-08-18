// Function to add a message to the chat display
function addMessage(sender, text, sources, save = true) {
    const chatMessages = document.querySelector('.chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);

    const messageText = document.createElement('div');
    messageText.classList.add('message-text');
    // Add a newline after each sentence to create a small gap
    messageText.textContent = text.replace(/\. /g, '.\n\n'); 
    messageDiv.appendChild(messageText);

    if (sources && sources.length > 0) {
        // Group sources by file name
        const sourcesByFile = sources.reduce((acc, current) => {
            const fileName = current.source;
            if (!acc[fileName]) {
                acc[fileName] = [];
            }
            acc[fileName].push(`Page ${current.page}`);
            return acc;
        }, {});

        // Format the sources string
        const sourcesDiv = document.createElement('div');
        sourcesDiv.classList.add('message-sources');

        let sourceString = "Sources:\n";
        for (const [fileName, pages] of Object.entries(sourcesByFile)) {
            sourceString += `- ${fileName} | Location: ${pages.join(', ')}\n`;
        }
        
        sourcesDiv.textContent = sourceString;
        messageDiv.appendChild(sourcesDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to the bottom

    // Save the new message to history only if the save flag is true
    if (save) {
        saveMessageToHistory({ sender, text, sources });
    }
}

// Function to handle sending a message
async function sendMessage() {
    const input = document.querySelector('.chat-input');
    const query = input.value.trim();
    if (!query) return;

    // Retrieve full chat history from local storage
    const chatHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];

    // Add user message to the chat window
    addMessage('user', query);
    input.value = ''; // Clear the input field

    // Add a temporary "typing..." message
    addTypingIndicator();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query, history: chatHistory }) // Send history with the query
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Remove typing indicator before adding the bot's message
        removeTypingIndicator();

        // Add the bot's response to the chat window
        addMessage('bot', data.answer, data.sources);

    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessage('bot', 'Sorry, I am unable to process your request at the moment.');
    }
}

// Function to add a "typing..." indicator
function addTypingIndicator() {
    const chatMessages = document.querySelector('.chat-messages');
    const typingIndicator = document.createElement('div');
    typingIndicator.classList.add('message', 'bot', 'typing-indicator');
    typingIndicator.textContent = '...';
    chatMessages.appendChild(typingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to remove the "typing..." indicator
function removeTypingIndicator() {
    const typingIndicator = document.querySelector('.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Function to display an introductory message
function showIntroductoryMessage() {
    // The intro message is now displayed but NOT saved to history
    const introMessage = `Hi there! I'm Sage, your virtual customer support assistant. How can I help you today?`;
    addMessage('bot', introMessage, null, false); // Do not save this message
}

// Function to display example queries
function displayExampleQueries() {
    const examplesDiv = document.createElement('div');
    examplesDiv.classList.add('example-queries-container');
    
    const exampleQueries = [
        "What is the return policy?",
        "How can I track my order?",
        "Do you offer assembly assistance?",
        "How do I care for wooden furniture?"
    ];

    exampleQueries.forEach(query => {
        const queryBtn = document.createElement('button');
        queryBtn.textContent = query;
        queryBtn.classList.add('example-query-btn');
        queryBtn.addEventListener('click', () => {
            document.querySelector('.chat-input').value = query;
            sendMessage();
        });
        examplesDiv.appendChild(queryBtn);
    });

    const chatMessages = document.querySelector('.chat-messages');
    chatMessages.appendChild(examplesDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Functions to manage chat history in local storage
function saveMessageToHistory(message) {
    let history = JSON.parse(localStorage.getItem('chatHistory')) || [];
    history.push(message);
    localStorage.setItem('chatHistory', JSON.stringify(history));
}

function loadChatHistory() {
    const history = JSON.parse(localStorage.getItem('chatHistory')) || [];
    if (history.length > 0) {
        history.forEach(msg => {
            // Do not save the message to history again when loading it
            const chatMessages = document.querySelector('.chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', msg.sender);

            const messageText = document.createElement('div');
            messageText.classList.add('message-text');
            messageText.textContent = msg.text;
            messageDiv.appendChild(messageText);

            if (msg.sources && msg.sources.length > 0) {
                const sourceDiv = document.createElement('div');
                sourceDiv.classList.add('message-sources');

                // Group sources by file name
                const sourcesByFile = msg.sources.reduce((acc, current) => {
                    const fileName = current.source;
                    if (!acc[fileName]) {
                        acc[fileName] = [];
                    }
                    acc[fileName].push(`Page ${current.page}`);
                    return acc;
                }, {});

                let sourceString = "Sources:\n";
                for (const [fileName, pages] of Object.entries(sourcesByFile)) {
                    sourceString += `- ${fileName} | Location: ${pages.join(', ')}\n`;
                }

                sourceDiv.textContent = sourceString;
                messageDiv.appendChild(sourceDiv);
            }
            chatMessages.appendChild(messageDiv);
        });
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}


// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    const chatIcon = document.querySelector('.chat-widget-container');
    const chatOverlay = document.querySelector('.chat-overlay');
    const closeBtn = document.querySelector('.close-btn');
    const sendBtn = document.querySelector('.send-btn');
    const chatInput = document.querySelector('.chat-input');
    const background = document.querySelector('.background-container');

    // NEW: Clear chat history and intro flag on page load
    localStorage.clear();

    // Open chat widget
    chatIcon.addEventListener('click', () => {
        chatOverlay.classList.add('visible');
        background.style.filter = 'blur(5px)';

        const hasSeenIntro = localStorage.getItem('hasSeenIntro');

        // Clear previous messages to avoid duplication
        document.querySelector('.chat-messages').innerHTML = '';
        
        // Show intro and example queries only on first open
        if (!hasSeenIntro) {
            showIntroductoryMessage();
            displayExampleQueries();
            localStorage.setItem('hasSeenIntro', 'true');
        } else {
            // Load history and display example queries on subsequent opens
            loadChatHistory();
            displayExampleQueries();
        }
    });

    // Close chat widget
    closeBtn.addEventListener('click', () => {
        chatOverlay.classList.remove('visible');
        background.style.filter = 'blur(0)';
    });
    
    // Send message on button click
    sendBtn.addEventListener('click', sendMessage);

    // Send message on Enter key press
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Dynamically adjust input area height (optional but good for UX)
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });
});