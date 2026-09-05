import json
import re
import sys
sys.dont_write_bytecode = True

from typing import List
from pydantic import BaseModel, Field

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function


RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
  """
  List of IDs of the applicants to retrieve resumes for
  """
  id_list: List[str] = Field(..., description="List of IDs of the applicants to retrieve resumes for")

class JobDescription(BaseModel):
  """
  Descriptions of a job to retrieve similar resumes for
  """
  job_description: str = Field(..., description="Descriptions of a job to retrieve similar resumes for") 



class RAGRetriever():
  def __init__(self, vectorstore_db, df):
    self.vectorstore = vectorstore_db
    self.df = df

  def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
    fused_scores = {}
    for doc_list in document_rank_list:
      for rank, (doc, _) in enumerate(doc_list.items()):
        if doc not in fused_scores:
          fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

  def __retrieve_docs_id__(self, question: str, k=50):
    docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
    docs_score = {str(doc.metadata["ID"]): score for doc, score in docs_score}
    return docs_score

  def retrieve_id_and_rerank(self, subquestion_list: list):
    document_rank_list = []
    for subquestion in subquestion_list:
      document_rank_list.append(self.__retrieve_docs_id__(subquestion, RAG_K_THRESHOLD))
    reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
    return reranked_documents

  def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
    id_resume_dict = dict(zip(self.df["ID"].astype(str), self.df["Resume"]))
    retrieved_ids = [doc_id for doc_id in sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True) if doc_id in id_resume_dict][:threshold]
    retrieved_documents = []
    for retrieved_id in retrieved_ids:
      retrieved_documents.append("Applicant ID " + retrieved_id + "\n" + id_resume_dict[retrieved_id])
    return retrieved_documents 
   


class SelfQueryRetriever(RAGRetriever):
  def __init__(self, vectorstore_db, df):
    super().__init__(vectorstore_db, df)

    self.prompt = ChatPromptTemplate.from_messages([
      ("system", "You are an expert in talent acquisition."),
      ("user", "{input}")
    ])
    self.meta_data = {
      "rag_mode": "",
      "query_type": "no_retrieve",
      "extracted_input": "",
      "subquestion_list": [],
      "retrieved_docs_with_scores": []
    }

  def retrieve_docs(self, question: str, llm, rag_mode: str):
    @tool(args_schema=ApplicantID)
    def retrieve_applicant_id(id_list: list):
      """Retrieve resumes for applicants in the id_list"""
      retrieved_resumes = []

      for id in id_list:
        try:
          resume_df = self.df[self.df["ID"].astype(str) == id].iloc[0][["ID", "Resume"]]
          resume_with_id = "Applicant ID " + resume_df["ID"].astype(str) + "\n" + resume_df["Resume"]
          retrieved_resumes.append(resume_with_id)
        except:
          return []
      return retrieved_resumes

    @tool(args_schema=JobDescription)
    def retrieve_applicant_jd(job_description: str):
      """Retrieve similar resumes given a job description"""
      subquestion_list = [job_description]

      if rag_mode == "RAG Fusion":
        subquestion_list += llm.generate_subquestions(question)
        
      self.meta_data["subquestion_list"] = subquestion_list
      retrieved_ids = self.retrieve_id_and_rerank(subquestion_list)
      self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
      retrieved_resumes = self.retrieve_documents_with_id(retrieved_ids)
      return retrieved_resumes
    
    def run_tool(tool_name: str, tool_input: dict):
      toolbox = {
        "retrieve_applicant_id": retrieve_applicant_id,
        "retrieve_applicant_jd": retrieve_applicant_jd
      }
      if tool_name not in toolbox:
        return None
      self.meta_data["query_type"] = tool_name
      self.meta_data["extracted_input"] = tool_input
      return toolbox[tool_name].run(tool_input)

    def parse_tool_call(response):
      if response is None:
        return None, None
      kwargs = getattr(response, "additional_kwargs", {}) or {}
      function_call = kwargs.get("function_call")
      if function_call:
        arguments = function_call.get("arguments") or "{}"
        if isinstance(arguments, str):
          arguments = json.loads(arguments or "{}")
        return function_call.get("name"), arguments
      tool_calls = kwargs.get("tool_calls") or getattr(response, "tool_calls", None) or []
      if tool_calls:
        tool_call = tool_calls[0]
        if isinstance(tool_call, dict):
          name = tool_call.get("name") or (tool_call.get("function") or {}).get("name")
          arguments = tool_call.get("args") or (tool_call.get("function") or {}).get("arguments") or {}
        else:
          name = getattr(tool_call, "name", None)
          arguments = getattr(tool_call, "args", None) or {}
        if isinstance(arguments, str):
          arguments = json.loads(arguments or "{}")
        return name, arguments
      return None, None

    self.meta_data["rag_mode"] = rag_mode
    openai_functions = [format_tool_to_openai_function(item) for item in [retrieve_applicant_id, retrieve_applicant_jd]]
    openai_tools = [{"type": "function", "function": function} for function in openai_functions]

    response = None
    try:
      bound_llm = llm.llm.bind(tools=openai_tools)
      response = bound_llm.invoke([
        SystemMessage(content="You are an expert in talent acquisition. If the user provides a job description or asks to find matching candidates, call retrieve_applicant_jd. If they provide applicant IDs, call retrieve_applicant_id."),
        HumanMessage(content=question),
      ])
    except Exception:
      try:
        llm_func_call = llm.llm.bind(functions=openai_functions)
        chain = self.prompt | llm_func_call | OpenAIFunctionsAgentOutputParser()
        parsed = chain.invoke({"input": question})
        if not isinstance(parsed, AgentFinish):
          result = run_tool(parsed.tool, parsed.tool_input)
          if isinstance(result, list):
            return result
      except Exception:
        response = None

    tool_name, tool_input = parse_tool_call(response)
    if tool_name:
      result = run_tool(tool_name, tool_input or {})
      if isinstance(result, list):
        return result

    applicant_ids = re.findall(r"(?:applicant\s+id\s+)([A-Za-z0-9._\-]+)", question, flags=re.IGNORECASE)
    if applicant_ids:
      result = run_tool("retrieve_applicant_id", {"id_list": applicant_ids})
      return result if isinstance(result, list) else []

    followup_pattern = re.compile(
      r"\b(compare|summarize|explain|tell me more|which (one|candidate)|top \d|rank them|why did|shortlist)\b",
      flags=re.IGNORECASE,
    )
    if followup_pattern.search(question) and len(question.split()) < 25:
      self.meta_data["query_type"] = "no_retrieve"
      return []

    result = run_tool("retrieve_applicant_jd", {"job_description": question})
    return result if isinstance(result, list) else []
