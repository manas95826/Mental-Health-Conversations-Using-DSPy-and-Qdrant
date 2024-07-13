from datasets import load_dataset
dataset = load_dataset('heliosbrahma/mental_health_chatbot_dataset')
df_pandas = dataset['train'].to_pandas()
documents = df_pandas['text'].to_list()
ids = list(range(1,len(documents)+1))


from qdrant_client import QdrantClient
client = QdrantClient("localhost", port= 6333)
client.delete_collection(collection_name= "Mental Health")
client.add(collection_name= "Mental Health", documents= documents, ids= ids)

import dspy

 Configure language model
llm = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.3')

 Create a Qdrant retriever model
from dspy.retrieve.qdrant_rm import QdrantRM
qdrant_retriever_model = QdrantRM("Mental Health", client, k=10)

 Configure DSPy settings
dspy.settings.configure(lm=llm, rm=qdrant_retriever_model)

class RAG(dspy.Module):
   def __init__(self, num_passages=3):
       super().__init__()
       self.retrieve = dspy.Retrieve(k=num_passages)
       self.generate_answer = dspy.ChainOfThought("context, question -> answer")

   def forward(self, question):
       context = self.retrieve(question).passages
       prediction = self.generate_answer(context=context, question=question)
       return dspy.Prediction(context=context, answer=prediction.answer)

class Coprocessor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rag = RAG()

    def forward(self, question):
         Retrieve relevant passages
        context = self.rag.retrieve(question).passages

         Generate a draft answer using the RAG
        draft_answer = self.rag.generate_answer(context=context, question=question).answer

         Refine the answer using a Coprocessor
        refined_answer = self.refine_answer(draft_answer, context)

        return dspy.Prediction(context=context, answer=refined_answer)

    def refine_answer(self, draft_answer, context):
         Implement your custom logic to refine the answer
         using the draft answer and the retrieved context
        refined_answer = draft_answer + " (Refined by Coprocessor)"
        return refined_answer

coprocessor = Coprocessor()
example_query = "Tell me about the panic attack?"
response = coprocessor(example_query)
print(response.answer)
