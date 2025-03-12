import streamlit as st
import pandas as pd
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate, LLMChain

# -------------------------------
# 1. Initialize the Chat Model
# -------------------------------
# We'll load the LLM in a cached resource function so it's not re-initialized on each run.
@st.cache_resource
def load_llm():
    """
    Initialize the ChatGoogleGenerativeAI model with your chosen parameters.
    Make sure you have your GOOGLE_API_KEY in st.secrets["GOOGLE_API_KEY"].
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",  # or another available model (e.g. "chat-bison@001")
        temperature=0,
        max_tokens=8000,
        api_key=st.secrets["GOOGLE_API_KEY"]
    )

llm = load_llm()

# -------------------------------
# 2. LLM-Driven Functions
# -------------------------------
def convert_business_to_technical(business_text):
    """Use the LLM to convert business requirements into technical requirements."""
    prompt_template = """Convert the following Business Requirements Document (BRD) into a detailed and structured Technical Requirements Document.
                        The output should be based solely on the information provided in the BRD—do not introduce any external details or hallucinations.
                        Include both functional and non-functional requirements for purchasing new servers, software, or any other technical assets.
                        Ensure the document is clear and unambiguous so that suppliers can easily understand the specifications.
                        BRD: {business_text}"""
    
    prompt = PromptTemplate(input_variables=["business_text"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(business_text=business_text)

def generate_rfp(technical_requirements):
    """Use the LLM to generate an RFP document from technical requirements."""
    prompt_template = """Convert the following Technical Requirements Document into a comprehensive and professional Request for Proposal (RFP) document.
                      The RFP should clearly articulate all technical details and performance criteria required from potential suppliers, based solely on the provided input.
                      Technical Requirements: {technical_requirements}
                      NOTE: do not introduce any external details or hallucinations."""
    
    prompt = PromptTemplate(input_variables=["technical_requirements"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(technical_requirements=technical_requirements)

def match_vendors(rfp_document, vendor_df):
    """
    Use the LLM to select top vendors based on rpf and vendor data.
    For simplicity, we only pass a subset of vendor data to the prompt.
    """
    # Add technical requirements from rfp and vendor evaluation criteria to the prompt
    vendor_data_str = vendor_df.head(10).to_csv(index=False)
    prompt_template =  """You have the following rfp:\n{rfp_document}\n\n
                        And here is a sample of the vendor data:\n{vendor_data}\n\n
                        Select the top 3 most suitable vendors. Return them in CSV format with columns: VendorName, KeyStrengths."""
    
    prompt = PromptTemplate(input_variables=["rfp_document", "vendor_data"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(
        rfp_document=rfp_document,
        vendor_data=vendor_data_str
    )

    try:
        shortlisted = pd.read_csv(io.StringIO(output))
    except Exception:
        st.error("Could not parse LLM output for vendor selection. Using fallback selection.")
        shortlisted = vendor_df.head(3)
    return shortlisted

def evaluate_bids(bids_df):
    """
    Use the LLM to evaluate bids and pick the top 2 based on price, quality, timelines, etc.
    Again, we only pass a sample of the bids to keep the prompt short.
    """
    bids_data_str = bids_df.head(10).to_csv(index=False)
    prompt_template = """You have the following bids:\n{bids_data}\n\n
                      Evaluate each bid based on price, quality, delivery timelines, and technology.
                      Select the top 2 bids and return them in CSV format with columns: BidID, EvaluationScore."""

    prompt = PromptTemplate(input_variables=["bids_data"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(bids_data=bids_data_str)
    try:
        evaluated = pd.read_csv(io.StringIO(output))
    except Exception:
        st.error("Could not parse LLM output for bid evaluation. Using fallback.")
        evaluated = bids_df.head(2)
    return evaluated

def simulate_negotiation_and_contract(top_bid):
    """
    Use the LLM to simulate a negotiation strategy and generate a contract draft from the top bid.
    """
    bid_details = "\n".join([f"{k}: {v}" for k, v in top_bid.items()])
    prompt_template = """You have the following top bid details:\n{bid_details}\n\n
                      First, outline a negotiation strategy. Then provide a draft contract. 
                      Separate the two with '---'."""

    prompt = PromptTemplate(input_variables=["bid_details"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(bid_details=bid_details)
    if "---" in output:
        negotiation_strategy, contract_draft = output.split("---", 1)
    else:
        negotiation_strategy = output
        contract_draft = "No contract draft found."
    return negotiation_strategy.strip(), contract_draft.strip()

# -------------------------------
# 3. Initialize Session State
# -------------------------------
if 'business_requirements' not in st.session_state:
    st.session_state['business_requirements'] = ''
if 'technical_requirements' not in st.session_state:
    st.session_state['technical_requirements'] = ''
if 'rfp_document' not in st.session_state:
    st.session_state['rfp_document'] = ''
if 'vendor_df' not in st.session_state:
    st.session_state['vendor_df'] = None
if 'bids_df' not in st.session_state:
    st.session_state['bids_df'] = None
if 'shortlisted_vendors' not in st.session_state:
    st.session_state['shortlisted_vendors'] = None
if 'evaluated_bids' not in st.session_state:
    st.session_state['evaluated_bids'] = None
if 'negotiation_strategy' not in st.session_state:
    st.session_state['negotiation_strategy'] = ''
if 'contract_draft' not in st.session_state:
    st.session_state['contract_draft'] = ''

# -------------------------------
# 4. Streamlit App Layout
# -------------------------------
st.set_page_config(page_title = "Transglobal Procurement Agent")
st.title("Procurement Agent")

# Step 1: Inputs
st.header("Step 1: Upload Inputs & Business Requirements")
with st.form("input_form"):
    business_text = st.text_area("Enter Business Requirements", height=150)
    vendor_file = st.file_uploader("Upload Vendor History CSV", type=["csv"])
    bids_file = st.file_uploader("Upload Bids CSV", type=["csv"])
    submit_disabled = not (business_text and vendor_file is not None and bids_file is not None)
    submitted_inputs = st.form_submit_button("Submit Inputs", disable = submit_disabled)

    if submitted_inputs:
        # Capture business requirements
        if business_text:
            st.session_state['business_requirements'] = business_text
            st.success("Business requirements captured.")
        else:
            st.error("Please enter business requirements.")
        
        # Process vendor CSV
        if vendor_file is not None:
            try:
                vendor_df = pd.read_csv(vendor_file)
                st.session_state['vendor_df'] = vendor_df
                st.success("Vendor CSV uploaded successfully.")
            except Exception as e:
                st.error(f"Error reading vendor CSV: {e}")
        else:
            st.error("Please upload Vendor History CSV.")
        
        # Process bids CSV
        if bids_file is not None:
            try:
                bids_df = pd.read_csv(bids_file)
                st.session_state['bids_df'] = bids_df
                st.success("Bids CSV uploaded successfully.")
            except Exception as e:
                st.error(f"Error reading Bids CSV: {e}")
        else:
            st.error("Please upload Bids CSV.")

# Step 2: Convert to Technical Requirements
st.header("Step 2: Convert Business to Technical Requirements")
if st.session_state['business_requirements']:
    if st.button("Convert to Technical Requirements"):
        tech_req = convert_business_to_technical(st.session_state['business_requirements'])
        st.session_state['technical_requirements'] = tech_req
        st.success("Generated Technical Requirements")
        with st.expander("Show Technical Requirements"):
            st.write(tech_req)
        # st.write("Generated Technical Requirements:")
        # st.text_area("Technical Requirements", value=tech_req, height=150)
else:
    st.info("Enter business requirements in Step 1.")

# Step 3: Generate RFP
st.header("Step 3: Generate RFP")
if st.session_state['technical_requirements']:
    if st.button("Generate RFP"):
        rfp = generate_rfp(st.session_state['technical_requirements'])
        st.session_state['rfp_document'] = rfp
        st.success("Generated RFP")
        with st.expander("Show RFP"):
            st.write(rfp)
        # st.text_area("RFP Document", value=rfp, height=150)
else:
    st.info("Please generate technical requirements in Step 2.")

# Step 4: Vendor Selection
st.header("Step 4: Vendor Selection")
if st.session_state['rfp_document'] and st.session_state['vendor_df'] is not None:
    if st.button("Select Vendors"):
        shortlisted = match_vendors(st.session_state['rfp_document'], st.session_state['vendor_df'])
        st.session_state['shortlisted_vendors'] = shortlisted
        st.success("Shortlisted Vendors")
        with st.expander("Show shortlisted vendors"):
            st.dataframe(shortlisted)
            # st.write(shortlisted)
        # st.write("Shortlisted Vendors:")
        # st.dataframe(shortlisted)
else:
    st.info("Ensure technical requirements are generated and vendor CSV is uploaded.")

# Step 5: Bid Evaluation
st.header("Step 5: Evaluate Bids")
if st.session_state['bids_df'] is not None:
    if st.button("Evaluate Bids"):
        evaluated = evaluate_bids(st.session_state['bids_df'])
        st.session_state['evaluated_bids'] = evaluated
        st.success("Evaluated Bids")
        with st.expander("Show Top Evaluated Bids"):
            st.dataframe(evaluated)
        # st.write("Top Evaluated Bids:")
        # st.dataframe(evaluated)
else:
    st.info("Please upload Bids CSV in Step 1.")

# Step 6: Negotiation & Contract
st.header("Step 6: Negotiation Simulation and Contract Drafting")
if st.session_state['evaluated_bids'] is not None and not st.session_state['evaluated_bids'].empty:
    top_bid = st.session_state['evaluated_bids'].iloc[0].to_dict()
    if st.button("Simulate Negotiation & Draft Contract"):
        negotiation_strategy, contract_draft = simulate_negotiation_and_contract(top_bid)
        st.session_state['negotiation_strategy'] = negotiation_strategy
        st.session_state['contract_draft'] = contract_draft
        st.success("Generated Negotiation Strategy and Contract Draft")
        with st.expander("Show Negotiation Strategy"):
            st.write(negotiation_strategy)
        # st.write("Negotiation Strategy:")
        # st.text_area("Negotiation Strategy", value=negotiation_strategy, height=100)
        with st.expander("Show Contract Draft"):
            st.write(contract_draft)
        # st.write("Contract Draft:")
        # st.text_area("Contract Draft", value=contract_draft, height=150)
else:
    st.info("Please evaluate bids in Step 5.")

# Step 7: Final Review & Downloads
st.header("Step 7: Final Review & Download")
if st.session_state['technical_requirements']:
    with st.expander("Show Technical Requirements"):
        st.write(st.session_state['technical_requirements'])

if st.session_state['rfp_document']:
    with st.expander("Show Request For Proposal"):
        st.write(st.session_state['rfp_document'])

if st.session_state['shortlisted_vendors'] is not None:
    with st.expander("Show shortlisted vendors"):
        st.dataframe(st.session_state['shortlisted_vendors'])
    
if st.session_state['evaluated_bids'] is not None:
    with st.expander("Show Top Evaluated Bids"):
        st.dataframe(st.session_state['evaluated_bids'])

if st.session_state['negotiation_strategy']:
    with st.expander("Show Negotiation Strategy"):
        st.write(st.session_state['negotiation_strategy'])

if st.session_state['contract_draft']:
    with st.expander("Show Contract Draft"):
        st.write(st.session_state['contract_draft'])

st.header("Download Final Documents")
if st.session_state['rfp_document']:
    st.download_button("Download RFP Document", st.session_state['rfp_document'], file_name="RFP_Document.txt")
if st.session_state['technical_requirements']:
    st.download_button("Download Technical Requirements", st.session_state['technical_requirements'], file_name="Technical_Requirements.txt")
if st.session_state['contract_draft']:
    st.download_button("Download Contract Draft", st.session_state['contract_draft'], file_name="Contract_Draft.txt")
