import { readFile } from "node:fs/promises";
import readline from "node:readline";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAI } from "openai";

const openai = new OpenAI(process.env.OPENAI_API_KEY);

const products = JSON.parse(await readFile("./products.json", "utf8"));

function createStore(products) {
    const embeddings = new OpenAIEmbeddings();
    return MemoryVectorStore.fromDocuments(
        products.map(
            (product) =>
                new Document({
                    pageContent: `Title: ${product.name}
  Description: ${product.description}
  Price: ${product.price}`,
                    metadata: { sourceId: product.id },
                })
        ),
        embeddings
    );
}

const store = await createStore(products);

async function searchProducts(query, count = 1) {
    const searchResults = await store.similaritySearch(query, count);
    return searchResults.map((result) =>
        products.find((product) => product.id === result.metadata.sourceId)
    );
}

async function generateResponse(message, chatHistory, context) {
    chatHistory.push({ role: "user", 
      content: message + `\n Context with related products: ${context}` });
    const response = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: chatHistory,
        temperature: 0,
        max_tokens: 4096,
    });
    const botMessage = response.choices[0].message.content;
    chatHistory.push({ role: "assistant", content: botMessage });
    return botMessage;
}



async function ragLoop() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const askQuestion = (query) =>
        new Promise((resolve) => rl.question(query, resolve));

    const chatHistory = [
        {
            role: "system",
            content: `You are helpful assistant to the product search chatbot!
            Please answer the user's questions about products.
            But do not answer questions that are not about products.
            If needed, you can ask the user for more information.
            Do not make up information, only answer questions about products that are in the database.`,
        }
    ];

    while (true) {
        const query = await askQuestion(
            'Welcome to gift store!\nPlease ask question about our products (type "exit" to quit):'
        );

        if (query.toLowerCase() === "exit") break;

        const products = await searchProducts(query, 3);

        let context = '';
        products.forEach((product, index) => {
            context +=
                `${product.name}: ${product.description}: ${product.price
                }`;
        });

        const response = await generateResponse(query, chatHistory, context);

        console.log(response);

    }

    rl.close();
}

await ragLoop();
