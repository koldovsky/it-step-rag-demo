import readline from "node:readline";
import OpenAI from "openai";

const openai = new OpenAI(process.env.OPENAI_API_KEY);

// const tools = [
//     {
//         "type": "function",
//         "function": {
//             "name": "bookCourse",
//             "description": "Books a course for the user.",
//             "parameters": {
//                 "type": "object",
//                 "properties": {
//                     "userName": {
//                         "type": "string",
//                         "description": "Name of the user.",
//                     },
//                     "courseName": {
//                         "type": "string",
//                         "enum": ["Python", "JavaScript", "AI"],
//                         "description": "Name of the course.",
//                     },
//                     "numOfSeats": {
//                         "type": "number",
//                         "description": "Number of seats",
//                     },
//                 },
//                 "required": ["userName", "courseName", "numOfSeats"],
//             },
//         }
//     },
// ];

async function generateResponse(message, chatHistory, tools) {
    chatHistory.push({ role: "user", content: message });
    const response = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: chatHistory,
        temperature: 0,
        max_tokens: 4096,
        // tools: tools,
        // tool_choice: 'bookCourse',
    });
    const botMessage = response.choices[0].message.content;
    chatHistory.push({ role: "assistant", content: botMessage });
    return botMessage;
}

async function chatLoop() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const askQuestion = (query) =>
        new Promise((resolve) => rl.question(query, resolve))


    const chatHistory = [
        {
            role: "system",
            content: `You are a helpful assistant on a IT programming school that delivers courses on AI, JavaScript, Python. 
                      Please answer only on the following topics: AI, JavaScript, Python. 
                      If user asks about other topics, please ask user to ask about AI, JavaScript, Python.
                      You can accept user requests to join specific courses according to schedule and available seats:
                      Python: 30th of May 2024 - 31th of August 2024 seats available - 3.
                      JavaScript: 30th of June 2024 - 30th of September 2024 seats available - 0.`
        }
    ];

    while (true) {
        const message = await askQuestion("Please ask question (type 'exit' to quit): ");
        if (message.toLowerCase() === "exit") break;
        const response = await generateResponse(message, chatHistory);
        console.log(`Assistant: ${response}`);
    }
    rl.close()
}

await chatLoop();
