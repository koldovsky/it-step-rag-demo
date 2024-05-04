import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { YoutubeLoader } from 'langchain/document_loaders/web/youtube'
import { CharacterTextSplitter } from 'langchain/text_splitter'
import { OpenAI } from "openai";

const openai = new OpenAI(process.env.OPENAI_API_KEY);

const question = process.argv[2] || 'hi'
const video = `https://youtu.be/PIEwl06O4rQ`

export const createStore = (docs) =>
    MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings())

export const docsFromYTVideo = async (video) => {
    const loader = YoutubeLoader.createFromUrl(video, {
        language: 'uk',
        addVideoInfo: true,
    })
    return loader.loadAndSplit(
        new CharacterTextSplitter({
            separator: ' ',
            chunkSize: 2500,
            chunkOverlap: 100,
        })
    )
}

const loadStore = async () => {
    const videoDocs = await docsFromYTVideo(video);

    return createStore(videoDocs)
}

const query = async () => {
    const store = await loadStore()
    const results = await store.similaritySearch(question, 1)
    console.log(results)

    const response = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo-16k-0613',
        temperature: 0,
        messages: [
            {
                role: 'assistant',
                content:
                    'You are a helpful AI assistant. Answser questions to your best ability.',
            },
            {
                role: 'user',
                content: `Answer the following question using the provided context. If you cannot answer the question with the context, don't lie and make up stuff. Just say you need more context.
        Question: ${question}
  
        Context: ${results.map((r) => r.pageContent).join('\n')}`,
            },
        ],
    })
    console.log(
        `Answer: ${response.choices[0].message.content}\n\nSources: ${results
            .map((r) => r.metadata.source)
            .join(', ')}`
    )
}

query()