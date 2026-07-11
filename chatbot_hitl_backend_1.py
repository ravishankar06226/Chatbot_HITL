import os
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Bio
from Bio.Seq import Seq
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
import gseapy as gp
import shutil


#Utility functions
def rev_comp(x):
    return str(Seq(x).reverse_complement())

import httplib2 as http
import json
from urllib.parse import urlparse

def get_hgnc_sym(x):
    headers = {
      'Accept': 'application/json',
    }

    uri = 'https://rest.genenames.org'
    path = '/search/'+str(x)

    target = urlparse(uri+path)
    method = 'GET'
    body = ''

    h = http.Http()

    response, content = h.request(
      target.geturl(),
      method,
      body,
      headers)

    if response['status'] == '200':
      # assume that content is a json reply
      # parse content with the json module
        data = json.loads(content)
        return str("|".join([i['symbol'] for i in data['response']['docs']]))
    else:
      return 'Not Found'

load_dotenv()

#model = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b",task="text-generation")
model = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",task="text-generation",max_new_tokens=200,do_sample=False,)
llm = ChatHuggingFace(llm=model)
# -------------------
# 1. LLM
# -------------------
#llm = ChatOpenAI()

# -------------------
# 2. Tools
# -------------------
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


@tool
def filter_DGE(state: Annotated[dict, InjectedState])->str:
    """filter gene : this tool can be used for filter differential expression genes based on pvalue 0.05 and log2foldcahnge +/-1 of a given file."""
    files_path=state.get("files")
    if files_path!=[]:
        try:
            df=pd.read_excel(files_path[0])
            df1=df[df['pvalue']<0.05]
            df1=df1[(df1['log2FoldChange']>=1) | (df1['log2FoldChange']<=-1)]
            df1.to_excel(files_path[0]+"_filter.xlsx",index=None)
            return {"message":"your filter DGE file is ready; it is always based on pvalue 0.05 and log2fc +/-1; please download"}
        except:
            return {"message":"given data is not in correct format; Please download the template using download button"}
    else:
        return {"message":"probably you forgot to attach file"}






@tool
def reverse_compliment(state: Annotated[dict, InjectedState])->str:
    """reverse complement : this tool can be used for reverse complement of Nucleotides sequence given in a file as list."""
    files_path=state.get("files")
    if files_path!=[]:
        try:
            df=pd.read_excel(files_path[0],header=None)
            df['reverse']=df.loc[:,0].apply(rev_comp)
            df.to_excel(files_path[0]+"_Rev.xlsx",index=None)
            return {"message":"your reverse compliment file is ready; please download"}
        except:
            return {"message":"given data is not in correct format; Please download the template using download button"}
    else:
        return {"message":"probably you forgot to attach file"}


@tool
def stacked_bar_plot(state: Annotated[dict,InjectedState])->str:
    """this tool can be used for plotting or generating a stacked bar plot. Also can be used for Alignment summary"""
    files_path=state.get("files")
    if files_path!=[]:
        try:
            df= pd.read_excel(files_path[0],index_col='Sample Name')
            df['Adapter(%)']= (df['Total Read']-df['Read After Adapter Trimming'])*100/df['Total Read']
            df['Contamination(%)']=(df['Read After Adapter Trimming']-df['Read After Contamination Removal'])*100/df['Total Read']
            df['% Read alignment'] =df['Read aligned']*100/df['Total Read']
            df['Unmapped(%)']= 100 - (df['Adapter(%)']+df['Contamination(%)']+df['% Read alignment'])
            df=df[['Adapter(%)','Contamination(%)','% Read alignment','Unmapped(%)']]
            ax=df.plot(kind='bar', stacked=True)
            plt.ylabel('Percentage of reads')
            plt.title('Alignment Matix')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(files_path[0]+"_stack.png", dpi=300, bbox_inches='tight')
            return {"message":"your stacked bar plot or alignment summary plot is ready. Please download using download button"}
        except:
            return {"message":"given data is not in correct format; Please download the template using download button"}
    else:
        return {"message":"Probably you forgot to attach the file"}

@tool
def hgnc_symbol(state: Annotated[dict,InjectedState])->str:
    """HGNC gene symbol, This tool can be used for fetching approved HGNC gene symbol from a list of gene provided in a file"""
    files_path=state.get("files")
    if files_path!=[]:
        try:
            df=pd.read_excel(files_path[0],header=None)
            df['HGNC_Approved_Symbol']=df.loc[:,0].apply(get_hgnc_sym)
            df.to_excel(files_path[0]+"_hgnc.xlsx",index=None)
            return {"message":"your HGNC symbol file is ready; please download"}
        except:
            return {"message":"given data is not in correct format; Please download the template using download button"}
    else:
        return {"message":"probably you forgot to attach file"}


@tool
def DGE(state:Annotated[dict,InjectedState],comparison_name:str)->str:
    """Run Differential Gene expression or DGE with given comparison name : DGE or diffential gene expression with a attached count file and given comparison name e.g control_vs_treated, control_vs_case, Control_vs_TEST or Test1_vs_Test2"""
    
    files_path=state.get("files")
    comparison=comparison_name
    if files_path!=[]:
        #try:
            df1=pd.read_excel(files_path[0],index_col=0)
            df2=pd.read_excel(files_path[1],index_col=0)
            df3=pd.read_excel(files_path[2],index_col=0)
            for df in [df1,df2,df3]:
                if list(df.columns)[0]=='Group':
                    metadata=df
                elif "Gene_Name" in list(df.columns):
                    annotation_df=df
                else:
                    counts_df=df
            metadata  = metadata[metadata['Group'].isin(comparison.split("_vs_"))]
            counts_df  = counts_df[list(metadata.index)]
            counts_df=counts_df.T
            if counts_df.index.equals(metadata.index):
                inference = DefaultInference(n_cpus=4)
                dds = DeseqDataSet(
                counts=counts_df,
                metadata=metadata,
                design="~Group",  # change to your variables
                #refit_cooks=True,
                inference=inference,)

                dds.deseq2()
                contrast=["Group",comparison.split("_vs_")[1],comparison.split("_vs_")[0] ]
                ds = DeseqStats(dds, contrast=["Group",comparison.split("_vs_")[1],comparison.split("_vs_")[0] ], inference=inference)
                ds.summary()
                df_full=ds.results_df
                df_full= pd.merge(annotation_df,df_full, how='inner', left_index=True, right_index=True)
                normalised_count_df= pd.DataFrame(dds.layers['normed_counts'], columns=counts_df.columns,index=counts_df.index)
                normalised_count_df=normalised_count_df.T
                df_full=pd.merge(df_full,normalised_count_df, how='inner', left_index=True, right_index=True)
                df_filter1=df_full[df_full['pvalue']<0.05]
                df_filter2=df_filter1[(df_filter1['log2FoldChange'] >=1) | (df_filter1['log2FoldChange']<=-1)]
                with pd.ExcelWriter(files_path[0]+'_DGE.xlsx', engine='openpyxl') as writer:
                    df_full.to_excel(writer, sheet_name='All', index=True,na_rep='NA')
                    df_filter2.to_excel(writer, sheet_name='p-value_0.05&LFC+-1', index=True,na_rep='NA')
                
                #Heatmap top 15 up and down genes
                df_up=df_filter2[df_filter2['log2FoldChange']>=1].sort_values(by='log2FoldChange',ascending=False ).head(15)
                df_down=df_filter2[df_filter2['log2FoldChange']<=-1].sort_values(by='log2FoldChange',ascending=False ).tail(15)
                df_final=pd.concat([df_up, df_down], axis=0)
                df_final.set_index('Gene_Name', inplace=True)
                df_final=df_final[list(metadata.index)]
                df_final=np.log2(df_final + 1)
                color_for_group = lambda x: "#3498db" if x==str(list(metadata["Group"])[0]) else "#e74c3c"
                metadata["Group"]=metadata["Group"].apply(color_for_group)
                condition_colors = metadata[["Group"]]

                g = sns.clustermap(
                df_final,
                method='ward',                # Linkage method for clustering
                metric='euclidean',           # Distance metric
                cmap='vlag',                  # Diverging colormap (blue = low, white = mean, red = high)
                center=0,                     # Explicitly center the colorbar at Z-score = 0
                col_colors=condition_colors,  # Adds condition tracking bars above columns
                linewidths=0.5,               # Adds thin lines between cells
                figsize=(8, 10),
                row_cluster=False,
                col_cluster=False)

                g.ax_heatmap.set_title('Differential Gene Expression Heatmap', fontsize=16, pad=20)
                g.ax_heatmap.set_xlabel('Samples', fontsize=12)
                g.ax_heatmap.set_ylabel('Differentially Expressed Genes', fontsize=12)
                g.cax.set_title('log2Normlised_Counts', fontsize=10) # Label the colorbar
                plt.savefig(files_path[0]+'_plot.png', dpi=300, bbox_inches='tight')
                return {"message":"your DGE or Differential gene expression and heatmap file is ready; please download. Note: this DGE is strictly based on pvalue<0.05 and log2FC +/-1"}
        #except:
         #   return {"message":"given data is not in correct format; Please download the template using download button"}
    else:
        return {"message":"probably you forgot to attach file"}




@tool
def pathway_analysis(state:Annotated[dict,InjectedState]):
    """Pathway analysis enrichment analysis KEGG pathway Reactome pathway GO term GO Annotation pathway analysis enrichment anaysis with the given species"""
    decision = interrupt(f"Please confirm the species? (human/mouse)")

    if isinstance(decision, str) and decision.lower().strip() == "human":
        files_path=state.get("files")

        if files_path!=[]:
            try:
                DEGs=pd.read_excel(files_path[0],1)
                UP_DEGs=DEGs[DEGs['log2FoldChange']>=1]['Gene_Name']
                DOWN_DEGs=DEGs[DEGs['log2FoldChange']<=-1]['Gene_Name']

                enr_GOBP_up = gp.enrichr(gene_list=UP_DEGs ,gene_sets=['GO_Biological_Process_2026'], organism='human',outdir=files_path[0].split(".")[0]+'/UPREGULATED_GOBP',cutoff=1 )
                enr_GOMF_up = gp.enrichr(gene_list=UP_DEGs,gene_sets=['GO_Molecular_Function_2026'],organism='human', outdir=files_path[0].split(".")[0]+'/UPREGULATED_GOMF',cutoff=1 )
                enr_GOCC_up = gp.enrichr(gene_list=UP_DEGs,gene_sets=['GO_Cellular_Component_2026'], organism='human', outdir=files_path[0].split(".")[0]+'/UPREGULATED_GOCC',cutoff=1 )
                enr_Reactome_up = gp.enrichr(gene_list=UP_DEGs,gene_sets=['Reactome_Pathways_2024'], organism='human',outdir=files_path[0].split(".")[0]+'/UPREGULATED_Reactome', cutoff=1 )
                enr_KEGG_up = gp.enrichr(gene_list=UP_DEGs,gene_sets=['KEGG_2026'], organism='human',outdir=files_path[0].split(".")[0]+'/UPREGULATED_KEGG', cutoff=1 )
                enr_GOBP_down = gp.enrichr(gene_list=DOWN_DEGs ,gene_sets=['GO_Biological_Process_2026'], organism='human',outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_GOBP',cutoff=1 )
                enr_GOMF_down = gp.enrichr(gene_list=DOWN_DEGs,gene_sets=['GO_Molecular_Function_2026'],organism='human', outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_GOMF',cutoff=1 )
                enr_GOCC_down = gp.enrichr(gene_list=DOWN_DEGs,gene_sets=['GO_Cellular_Component_2026'], organism='human', outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_GOCC',cutoff=1 )
                enr_Reactome_down = gp.enrichr(gene_list=DOWN_DEGs,gene_sets=['Reactome_Pathways_2024'], organism='human',outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_Reactome', cutoff=1 )
                enr_KEGG_down = gp.enrichr(gene_list=DOWN_DEGs,gene_sets=['KEGG_2026'], organism='human',outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_KEGG', cutoff=1 )

                shutil.make_archive(files_path[0]+"_Enrichment","zip", files_path[0].split(".")[0])
                return {"message":"your DGE pathway analysis is ready, please downoad using download button"}
            except:
                shutil.copy("static/ForPathwayAnalysis.xlsx",files_path[0]+"_Template.xlsx")
                return {"message":"given data is not in correct format; Please download the template using download button"}
        else:
            return {"message":"probably you forgot to attach file"}
    else:
        files_path=state.get("files")
        if files_path!=[]:
            try:
                DEGs=pd.read_excel(files_path[0],1)
                UP_DEGs=DEGs[DEGs['log2FoldChange']>=1]['Gene_Name']
                DOWN_DEGs=DEGs[DEGs['log2FoldChange']<=-1]['Gene_Name']

                enr_GOBP_up = gp.enrichr(gene_list=UP_DEGs ,gene_sets=['GO_Biological_Process_2026'], organism='mouse',outdir=files_path[0].split(".")[0]+'/UPREGULATED_GOBP',cutoff=1 )
                enr_GOMF_up = gp.enrichr(gene_list=UP_DEGs,gene_sets=['GO_Molecular_Function_2026'],organism='mouse', outdir=files_path[0].split(".")[0]+'/UPREGULATED_GOMF',cutoff=1 )
                enr_GOCC_up = gp.enrichr(gene_list=UP_DEGs,gene_sets=['GO_Cellular_Component_2026'], organism='mouse', outdir=files_path[0].split(".")[0]+'/UPREGULATED_GOCC',cutoff=1 )
                enr_Reactome_up = gp.enrichr(gene_list=UP_DEGs,gene_sets=['Reactome_Pathways_2024'], organism='mouse',outdir=files_path[0].split(".")[0]+'/UPREGULATED_Reactome', cutoff=1 )
                enr_KEGG_up = gp.enrichr(gene_list=UP_DEGs,gene_sets=['KEGG_Pathways_2026'], organism='mouse',outdir=files_path[0].split(".")[0]+'/UPREGULATED_KEGG', cutoff=1 )
                enr_GOBP_down = gp.enrichr(gene_list=DOWN_DEGs ,gene_sets=['GO_Biological_Process_2026'], organism='mouse',outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_GOBP',cutoff=1 )
                enr_GOMF_down = gp.enrichr(gene_list=DOWN_DEGs,gene_sets=['GO_Molecular_Function_2026'],organism='mouse', outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_GOMF',cutoff=1 )
                enr_GOCC_down = gp.enrichr(gene_list=DOWN_DEGs,gene_sets=['GO_Cellular_Component_2026'], organism='mouse', outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_GOCC',cutoff=1 )
                enr_Reactome_down = gp.enrichr(gene_list=DOWN_DEGs,gene_sets=['Reactome_Pathways_2024'], organism='mouse',outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_Reactome', cutoff=1 )
                enr_KEGG_down = gp.enrichr(gene_list=DOWN_DEGs,gene_sets=['KEGG_2026'], organism='mouse',outdir=files_path[0].split(".")[0]+'/DOWNREGULATED_KEGG', cutoff=1 )

                shutil.make_archive(files_path[0]+"_Enrichment","zip", files_path[0].split(".")[0])
                return {"message":"your DGE pathway analysis is ready, please downoad using download button"}
            except:
                shutil.copy("static/ForPathwayAnalysis.xlsx",files_path[0]+"_Template.xlsx")
                return {"message":"given data is not in correct format; Please download the template using download button"}
        else:
            return {"message":"probably you forgot to attach file"}





tools = [get_stock_price, purchase_stock, reverse_compliment, stacked_bar_plot, filter_DGE, hgnc_symbol,DGE,pathway_analysis]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    files:list[str]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    system_message="""You are a strict conversational assistant. You are ONLY allowed to answer questions using the information provided by the tools available to you.Do not use your own pre-trained knowledge to answer questions. If the information cannot be found using your tools, or if you do not have a tool for the question, respond with exactly: "I am sorry, but I can only answer questions using my configured tools."""
    messages = [SystemMessage(content=system_message)]+state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer (in-memory)
# -------------------
memory = MemorySaver()

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)
