# Objective: To build an Agentic AI that automates TransGlobal Industries' procurement process by leveraging LLMs, LangChain, and Streamlit. 

import streamlit as st
import os
import langchain
import pandas as pd
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# displaying versions of libraries used
print("streamlit version:", st.__version__)
print("langchain version:", langchain.__version__)
print("pandas version:", pd.__version__)

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
        api_key=os.environ.get("GOOGLE_API_KEY")
    )

llm = load_llm()

# -------------------------------
# 2. LLM-Driven Functions
# -------------------------------
def brd_to_trd(brd):
    """Use the LLM to convert business requirements into technical requirements."""
    prompt_template = """You are a Senior Technical Procurement Specialist at a manufacturing firm. Your task is to convert the following Business Requirements Document (BRD) into a comprehensive, strictly technical requirements document. Use only the details provided in the BRD and do not add any external information. The output must include both functional and non-functional requirements for purchasing new servers, software, or other technical assets, and it should be written so that suppliers can unambiguously understand every specification.
                        Your output must clearly address each of the following key factors and include specific examples or ranges where applicable:
                        Requirement Clarity & Specification Accuracy
                        Example: If the BRD specifies intensive software handling, include a statement like "System must include 16 GB DDR4 RAM or more" to support heavy multitasking.
                        Example: For high computational performance, specify a processor such as "Intel Core i7 (11th Gen) or equivalent" or higher.
                        Standardized Descriptions
                        Example: Instead of vague terms like "fast processor," clearly state "Processor: Intel Core i7 (11th Gen) or better."
                        Example: For storage, specify "512GB SSD or higher" if the BRD indicates high storage needs.
                        Hidden Costs Consideration
                        Example: Include notes like "Include considerations for logistics, customs, and storage costs" when the BRD mentions any potential hidden costs.
                        Example: If delivery or installation is mentioned, ensure to note any additional expenses that might be incurred.
                        Regulatory Compliance
                        Example: If the BRD refers to adhering to industry standards, include "Ensure compliance with import/export laws and applicable industry standards."
                        Example: Specify required certifications or licenses if mentioned (e.g., "Operating System must be a licensed version of Windows 11 Pro").
                        Risk Mitigation
                        Example: If the BRD highlights potential risks, output "Include contract clauses for warranties (e.g., 3 years onsite warranty), penalties for delays, and risk mitigation measures."
                        Example: Mention hidden risks like "Consideration for potential downtime and its impact on operations."
                        Additionally, use the following example vendor bid to guide the level of detail expected. This example is for reference only and should not be copied verbatim unless the BRD contains matching details:
                        Processor: Intel Core i7 (11th Gen) or equivalent
                        RAM: 16 GB DDR4 or more
                        Storage: 512GB SSD or higher
                        Display: 15.6-inch FHD (1920×1080) IPS
                        Graphics: Integrated or discrete as specified
                        Battery Life: 10 hours or more
                        Ports: e.g., 2× USB 3.0, 1× USB Type-C, HDMI, 3.5mm audio jack, Ethernet
                        Connectivity: Wi-Fi 6, Bluetooth 5.0
                        Operating System: Windows 11 Pro (licensed)
                        Keyboard: Backlit keyboard
                        Weight: Approximately 2.2 kg
                        Warranty: 3 years onsite warranty
                        Software: Pre-installed Windows 11 Pro with license, Office 365 Business, Antivirus with a 3-year subscription
                        Financial Proposal: e.g., Unit Price: $1,200 per unit; total cost for 100 units: $120,000; additional software/services: $5,000; Payment Terms: 50% advance, 50% on delivery
                        Instruction:
                        Convert the provided BRD into a detailed Technical Requirements Document that strictly contains technical details extracted from the BRD. 
                        For each key technical aspect (e.g., processor, RAM, storage), if the BRD suggests intensive or high-performance needs, the output must specify exact or minimum values such as "16 GB DDR4" or "512GB SSD or more" as appropriate. Similarly, include specific requirements for connectivity, operating system, software, hidden cost considerations, regulatory compliance, and risk mitigation measures.
                        BRD: {brd}"""
    
    prompt = PromptTemplate(input_variables=["brd"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(brd = brd)

def trd_to_rfp(trd):
    """Use the LLM to generate an RFP document from technical requirements."""
    prompt_template = """Convert the following Technical Requirements Document into a comprehensive and professional Request for Proposal (RFP) document.
                      The RFP should clearly articulate all technical details and performance criteria required from potential suppliers, based solely on the provided input.
                      Technical Requirements: {trd}
                      NOTE: do not introduce any external details or hallucinations."""
    
    prompt = PromptTemplate(input_variables=["trd"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(trd = trd)

def match_vendors(vendor_df):
    """
    Use the LLM to select top vendors based on rpf and vendor data.
    For simplicity, we only pass a subset of vendor data to the prompt.
    """
    # Add technical requirements from rfp and vendor evaluation criteria to the prompt
    # Convert the vendor DataFrame to a CSV string
    vendor_csv_str = vendor_df.to_csv(index=False)
    # Acceptance criteria: Quality_of_Goods: 0.4, Delivery_punctuality: 0.35,  Contract_term_compliance: 0.25
    prompt_template =  """You are provided with a vendor dataset as input (variable: {vendor_data}) where each row represents a single delivery. The dataset includes the following key attributes:

                        Vendor_name: Name of the vendor.
                        Delivery_punctuality: Delivery punctuality score (1-10, with 10 being the best).
                        Quality_of_goods: Quality score for delivered goods (1-10, with 10 being the best).
                        Contract_term_compliance: Contract compliance score (1-10, with 10 being the best).
                        
                        Follow these steps:
                        
                        1. Aggregate Data: Group the dataset by 'Vendor_name'. For each vendor, calculate the mean values for 'Delivery_punctuality', 'Quality_of_goods', and 'Contract_term_compliance'. For example, for Vendor A, determine:
                        
                        Mean_Delivery_punctuality_Vendor_A
                        Mean_Quality_of_goods_Vendor_A
                        Mean_Contract_term_compliance_Vendor_A
                        
                        2. Compute Weighted Average: For each vendor, compute the weighted average score using the weights:
                        
                        w1 = 0.4 for 'Quality_of_goods'
                        w2 = 0.35 for 'Delivery_punctuality'
                        w3 = 0.25 for 'Contract_term_compliance'
                        Specifically, for a vendor X:
                        Weighted_Average_X = (Mean_Quality_of_goods_X * 0.4) + (Mean_Delivery_punctuality_X * 0.35) + (Mean_Contract_term_compliance_X * 0.25)
                        
                        3. Select Top Vendors: Rank the vendors based on their weighted average scores in descending order and choose the top 3.
                        
                        4. Output Format: Return a list of the top 3 vendors.
                        
                        Remember, Use only the information provided in the vendor dataset. Do not introduce any external data or hallucinate details.
                        Do not return python code.
                        Output Format:
                        Strictly return only the CSV output with column: VendorName
                        Do not return any other strings or text"""
    
    prompt = PromptTemplate(input_variables=["vendor_data"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(vendor_data=vendor_csv_str)
    # Clean up output to remove extra blank lines or unwanted text
    output = output.strip()
    shortlisted = pd.read_csv(io.StringIO(output))
    return shortlisted

def generate_tender_doc(trd):
    prompt_template = """ Using the provided Technical Requirements Document (TRD) (variable: {trd}), generate a professional tender document for procurement purposes. The tender document should be structured, clear, and concise, ensuring that vendors fully understand the technical and business requirements.
                        The tender document should include the following sections:
                        
                        1.	Title & Issuing Organization: Clearly state the tender title and the organization issuing it.
                        2.	Invitation to Tender: A brief introduction explaining the purpose of the tender and the procurement scope.
                        3.	Scope of Work: A summary of vendor responsibilities, including product delivery, pre-installation requirements, and support expectations.
                        4.	Technical Requirements: 
                            o	Hardware specifications (processor, RAM, storage, display, connectivity, etc.).
                            o	Software requirements (pre-installed OS, applications, security software).
                            o	Security and compliance requirements (e.g., TPM, biometric authentication).
                        5.	Business Requirements: 
                            o	Quantity of units required.
                            o	Pricing and expected bulk purchase discounts.
                            o	Payment terms and preferred conditions.
                            o	Warranty and support expectations.
                            o	Delivery timelines and shipping requirements.
                            o	Packaging and inventory tagging needs.
                        6.	Proposal Submission Guidelines: 
                            o	Instructions on submission format (e.g., PDF).
                            o	Required documents (e.g., company profile, compliance statement, pricing breakdown).
                            o	Submission deadline and contact details.
                        7.	Closing Statement: A formal closing encouraging vendors to submit proposals and reinforcing the deadline.
                        
                        Ensure that the document is professionally formatted with section headers, bullet points where necessary, and a business-appropriate tone. The final output should be structured in a way that makes it easy for vendors to review and respond efficiently.
                        Remember, Use only the information provided in the TRD. Do not introduce any external data or hallucinate details."""
    
    prompt = PromptTemplate(input_variables=["trd"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(trd = trd)

def generate_email(rfp):
    prompt_template = """ Generate a professional email addressed to the shortlisted vendors inviting them to submit their bids for the attached Tender Document and Request for Proposal (RFP). The email should be formal, clear, and concise, ensuring that vendors understand the expectations and submission requirements.
                        Check for the name of Issuing organization from the attached RFP document (Variable: {rfp})
                        The email should include only the following elements:
                        1.	Subject Line: A professional subject line indicating an invitation to submit a bid.
                        2.	Salutation: Address the vendors respectfully.
                        3.	Introduction: Briefly introduce the issuing organization and the purpose of the email.
                        4.	Invitation to Bid: Clearly state that the vendor has been shortlisted and is invited to submit a proposal based on the attached documents.
                        5.	Attachments: Mention the attached Tender Document and RFP for their reference.
                        6.	Closing Statement: Encourage timely submissions and express anticipation for their response.
                        7.	Signature: Include the sender’s name, designation, and contact details.
                        Ensure the email is concise, professional, and engaging, while maintaining a clear call to action for the vendors to submit their bids on time."""
  
    prompt = PromptTemplate(input_variables=["rfp"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(rfp = rfp)

def evaluate_bids(bids_df, trd):
    """
    Use the LLM to evaluate bids and pick the top 2 based on price, quality, timelines, etc.
    Again, we only pass a sample of the bids to keep the prompt short.
    """
    bids_csv_text = bids_df.to_string(index=False)  # Converts the DataFrame to a text

    prompt_template = """Chain-of-thought prompt:

                        1. Read the bids file {bids_csv_text} and the Technical Requirements Document {trd}.
                        2. For each bid, extract the Unit Price and rank bids so that the bid with the lowest Unit Price gets the highest numerical rank (e.g., if there are 4 bids, the lowest price gets rank 4 and the highest gets rank 1). Record this as price_BidX for each bid.
                        3. For each bid, compare its technical proposal attributes (processor, RAM, storage, display, graphics, battery life, ports, connectivity, operating system, and warranty) with the specifications in the TRD. Assign a technical capability score (tech_cap_BidX) from 0 to 1 based on the similarity.
                        4. For each bid, extract the Lead Time from the delivery schedule and rank bids so that the bid with the lowest Lead Time gets the highest numerical rank and the bid with the highest Lead Time gets rank 1. Record this as delivery_BidX.
                        5. Independently rank the overall quality of each bid (excluding technical specifications) so that the bid with the best quality gets the highest numerical rank and the worst quality gets rank 1. Record this as quality_BidX.
                        6. Compute a weighted average score for each bid using the formula:  
                          Weighted_Average_BidX = (price_BidX * 0.4) + (tech_cap_BidX * 0.3) + (quality_BidX * 0.2) + (delivery_BidX * 0.1).
                        7. Rank all bids by their weighted average scores in descending order and select the top 2 bids.
                        8. Output only a CSV file with a single column "VendorName" listing the vendor names of the top 2 bids.
                        
                        Return strictly the CSV output with no additional text."""

    prompt = PromptTemplate(input_variables=["bids_csv_text", "trd"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(bids_csv_text = bids_csv_text, trd = trd)
    output = output.strip()
    shortlisted = pd.read_csv(io.StringIO(output))
    return shortlisted

def simulate_negotiation_and_contract(top_bid, bids_df):
    """
    Use the LLM to simulate a negotiation strategy and generate a contract draft from the top bid.
    """
    # Create a multi-line string of the top bid's details by formatting each key-value pair as "key: value" and joining them with newline characters.
    top_bids_str = "\n".join([f"{k}: {v}" for k, v in top_bid.items()])
    # Converts the DataFrame to a text
    bids_csv_text = bids_df.to_string(index=False)  
    
    prompt_template_negotiation = """You are a Procurement Negotiator.
                        First, you will check the names of the shortlisted bids in the file {top_bids}.
                        Store the name of the first bid from the shortlisted bids as "TopBid" (this is only for your reference, do not mention "TopBid" in the response)
                        To proceed further you will only consider the details of these shortlisted bids from the file {bids_details}. now, follow the intructions below:
                        Outline a robust negotiation strategy. Apart from other vital things, construct the negotiation strategy including the following factors also:
                            1. BATNA: Analyze the pricing of bids to determine the company's Best Alternative to a Negotiated Agreement (BATNA). Evaluate alternatives, given the shortlisted bids.
                            2. Simulate negotiation scenarios to devise robust negotiation strategies for engaging with the preferred supplier.
                            3. Market Trends, Supplier Pricing and Bulk Discounts: Your recommendations should ensure that the procuring company is well-prepared to secure favorable terms by leveraging competitive market trends, supplier pricing, and potential bulk discounts.
                            4. Benchmarking: Compare prices across vendors.
                            5. Using first principles thinking, break down the negotiation challenge into its fundamental components. Identify the core drivers—such as supplier cost structures, market trends, and value determinants—without relying on conventional assumptions.
                            6. Leverage Competition: Use the competitive environment to negotiate better terms.
                        Limit your response to no more than 400 words
                        """
    prompt_negotiation = PromptTemplate(input_variables=["top_bids", "bids_details"], template=prompt_template_negotiation)
    chain_negotiate = LLMChain(llm=llm, prompt=prompt_negotiation)
    output_negotiate = chain_negotiate.run(top_bids = top_bids_str, bids_details = bids_csv_text)

    
    prompt_template_risk = """You are a risk manager, expert in identifying potential risks associated with supplier relationships during procurement activities.
                            First, you will check the names of the shortlisted bids in the file {top_bids}.
                            Store the name of the first bid from the shortlisted bids as "TopBid" (this is only for your reference, do not mention "TopBid" in the response)
                            To proceed further you will only consider the details of these shortlisted bids from the file {bids_details}. Now, follow the intructions below:
                            Carefully analyze the functional and non-functional characteristics of the "TopBid", including its technical specifications (e.g., processor, RAM, storage, performance benchmarks), operational features, and quality parameters.
                            Assess potential risks such as supplier reliability and financial stability, compliance with regulatory and industry standards, and adherence to warranty and service level agreements.
                            Additionally, evaluate hidden costs (logistics, maintenance, support), potential contract vulnerabilities, delivery timelines, market conditions, and any discrepancies in vendor performance history.
                            Generate a detailed risk assessment report that highlights critical risk factors, their potential impact, and recommended mitigation strategies.
                            Limit your response to no more than 450 words
                            """
    prompt_risk = PromptTemplate(input_variables=["top_bids", "bids_details"], template = prompt_template_risk)
    chain_risk = LLMChain(llm = llm, prompt = prompt_risk)
    output_risk = chain_risk.run(top_bids = top_bids_str, bids_details = bids_csv_text)

    
    prompt_template_contract = """You are an experienced Procurement Manager specializing in contract creation, with deep expertise in the legal aspects of procurement agreements. 
                                First, you will check the names of the shortlisted bids in the file {top_bids}.
                                Store the name of the first bid from the shortlisted bids as "TopBid" (this is only for your reference, do not mention "TopBid" in the response)
                                To proceed further you will only consider the details of these shortlisted bids from the file {bids_details}. Now, follow the intructions below:
                                Your task is to draft a comprehensive contract document exclusively for 'TopBid,' incorporating key findings from the risk assessment report {risk_report}.
                                The contract should be legally sound, ensuring clear and enforceable commitments between both parties. 
                                It must include robust clauses for risk mitigation, performance guarantees, and dispute resolution to safeguard the interests of all stakeholders.

                                Key Components to Include:
                                1. Scope of Work (SOW): Clearly define the goods/services being procured.
                                2. Pricing & Payment Terms: Specify total cost, discounts, payment schedules, and penalties for late payments.
                                3. Service Level Agreements (SLAs): Establish performance expectations, response times, and penalties for non-compliance.
                                4. Warranties & Support: Outline warranty periods, service coverage, and response times for support.
                                5. Compliance & Legal Risks: Ensure adherence to data security, regulatory requirements, and intellectual property protections.
                                6. Termination & Liability: Define exit clauses, liabilities, indemnities, and dispute resolution mechanisms.
                                
                                Ensure that the contract maintains a professional tone, aligns with industry best practices, and mitigates potential risks identified in the assessment report.
                                """
    prompt_contract = PromptTemplate(input_variables=["top_bids", "bids_details", "risk_report"], template = prompt_template_contract)
    chain_contract = LLMChain(llm = llm, prompt = prompt_contract)
    output_contract = chain_contract.run(top_bids = top_bids_str, bids_details = bids_csv_text, risk_report = output_risk) 
    
    # Split the output into parts using '---' as the delimiter.
    # If there are at least 3 parts, assign them to negotiation_strategy, risk_assessment, and contract_draft.
    # Otherwise, assign fallback messages for any missing parts.

    return output_negotiate, output_risk, output_contract

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
if 'tender_doc' not in st.session_state:
    st.session_state['tender_doc'] = ''
if 'email' not in st.session_state:
    st.session_state['email'] = ''
if 'bids_df' not in st.session_state:
    st.session_state['bids_df'] = None
if 'shortlisted_vendors' not in st.session_state:
    st.session_state['shortlisted_vendors'] = None
if 'evaluated_bids' not in st.session_state:
    st.session_state['evaluated_bids'] = None
if 'negotiation_strategy' not in st.session_state:
    st.session_state['negotiation_strategy'] = ''
if 'risk_assessment' not in st.session_state:
    st.session_state['risk_assessment'] = ''
if 'contract_draft' not in st.session_state:
    st.session_state['contract_draft'] = ''

# -------------------------------
# 4. Streamlit App Layout
# -------------------------------
st.set_page_config(page_title = "Procurement Agent", page_icon = "📦",initial_sidebar_state="expanded")
st.title("Transglobal Procurement Agent")

# Step 1: Inputs
st.header("Step 1: Enter Business Requirements")
with st.form("input_form"):
    business_text = st.text_area("Text area to enter business requirements", height=150)
    submitted_inputs = st.form_submit_button("Submit BRD")

    if submitted_inputs:
        if business_text:
            st.session_state['business_requirements'] = business_text
            st.success("Business requirements captured.")            
        else:
            st.error("Please enter business requirements.")
        
# Step 2: Convert to Technical Requirements
st.header("Step 2: Convert BRD to Technical Requirements Document")
if st.session_state['business_requirements']:
    if st.button("Convert to Technical Requirements"):
        with st.spinner('Generating Technical Requirements Document'):
                trd = brd_to_trd(st.session_state['business_requirements'])
        # trd = brd_to_trd(st.session_state['business_requirements'])
        st.session_state['technical_requirements'] = trd
        st.success("Generated Technical Requirements")
        with st.expander("Show Technical Requirements"):
            st.write(trd)
else:
    st.info("Ensure that Business Requirements are captured in Step 1")

# Step 3: Generate RFP
st.header("Step 3: Generate RFP from TRD")
if st.session_state['technical_requirements']:
    if st.button("Generate RFP"):
        with st.spinner('Generating Request for Proposal'):
                rfp = trd_to_rfp(st.session_state['technical_requirements'])        
        st.session_state['rfp_document'] = rfp
        st.success("Generated Request for Proposal")
        with st.expander("Show RFP"):
            st.write(rfp)
else:
    st.info("Please generate Technical Requirements in Step 2")

# Step 4: Vendor Selection
st.header("Step 4: Vendor Selection")
if st.session_state['rfp_document']:
    vendor_file = st.file_uploader("Upload Vendor History CSV", type=["csv"])
    if st.button("Select Vendors"):
        if vendor_file is not None:
            try:
                vendor_df = pd.read_csv(vendor_file)
                st.session_state['vendor_df'] = vendor_df
                st.success("Vendor CSV uploaded successfully.")
                with st.spinner('Please wait, vendor selection process is going on...'):
                    shortlisted = match_vendors(st.session_state['vendor_df'])
                st.session_state['shortlisted_vendors'] = shortlisted
                st.success("Shortlisted Vendors")
                with st.expander("Show shortlisted vendors"):
                    st.dataframe(shortlisted)
            except Exception as e:
                st.error(f"Error reading vendor CSV: {e}")        
        else:
            st.error("Please upload Vendor History CSV.")        
else:
    st.info("Ensure RFP is generated in Step 3")

# Step 5: Producing a tender document and generating email for the shortlisted vendors
st.header("Step 5: Producing Tender Document and Generating Emails for vendors")
if st.session_state['shortlisted_vendors'] is not None:
    if st.button("Generate Tender Document"):
        with st.spinner('Generating Tender Document...'):
            tender_doc = generate_tender_doc(st.session_state['technical_requirements'])
        st.session_state['tender_doc'] = tender_doc
        st.success("Generated Tender Document")
        with st.expander("Show Tender Document"):
            st.write(tender_doc)
   
    if st.session_state['tender_doc']:
        if st.button("Generate Email for shortlisted Vendors"):
            with st.spinner('Generating Email...'):
                email = generate_email(st.session_state['rfp_document'])
            st.session_state['email'] = email
            st.success("Generated Email for vendors")
            with st.expander("Show Email"):
                st.write(email)
    else:
        st.info("Ensure Tender Document is generated")
else:
    st.info("Ensure shortlisted vendors list is generated in step 4")


# Step 6: Bid Evaluation
st.header("Step 6: Evaluate Bids")
if st.session_state['email']:
    bids_file = st.file_uploader("Upload Bids CSV", type=["csv"])
    if st.button("Evaluate Bids"):
        if bids_file is not None:
            try:
                bids_df = pd.read_csv(bids_file)
                st.session_state['bids_df'] = bids_df
                st.success("Bids CSV uploaded successfully")
                with st.spinner('Evaluating Bids...'):
                    evaluated = evaluate_bids(st.session_state['bids_df'], st.session_state['technical_requirements'])
                st.session_state['evaluated_bids'] = evaluated
                st.success("Evaluated Bids")
                with st.expander("Show Top Evaluated Bids"):
                    st.dataframe(evaluated)
            except Exception as e:
                st.error(f"Error reading bids CSV: {e}")
    
else:
    st.info("Ensure Email for vendors is generated")

# Step 7: Negotiation & Contract
st.header("Step 7: Generating Negotiation Strategy, Risk Assessment and Contract Document")
if st.session_state['evaluated_bids'] is not None and not st.session_state['evaluated_bids'].empty:
    top_bid = st.session_state['evaluated_bids'].iloc[0].to_dict()
    if st.button("Simulate Negotiation & Draft Contract"):
        negotiation_strategy, risk_assessment, contract_draft = simulate_negotiation_and_contract(top_bid, st.session_state['bids_df'])
        st.session_state['negotiation_strategy'] = negotiation_strategy
        st.session_state['risk_assessment'] = risk_assessment
        st.session_state['contract_draft'] = contract_draft
        st.success("Generated Negotiation Strategy, Risk Assessment Report and Contract Draft")
        with st.expander("Show Negotiation Strategy"):
            st.write(negotiation_strategy)
        with st.expander("Show Risk Assessment Report"):
            st.write(risk_assessment)
        with st.expander("Show Contract Draft"):
            st.write(contract_draft)
else:
    st.info("Please evaluate bids in Step 6")

# Step 8: Final Review & Downloads
st.header("Step 8: Final Review & Download")
if st.session_state['technical_requirements']:
    with st.expander("Show Technical Requirements"):
        st.write(st.session_state['technical_requirements'])

if st.session_state['rfp_document']:
    with st.expander("Show Request For Proposal"):
        st.write(st.session_state['rfp_document'])

if st.session_state['shortlisted_vendors'] is not None:
    with st.expander("Show shortlisted vendors"):
        st.dataframe(st.session_state['shortlisted_vendors'])

if st.session_state['tender_doc']:
    with st.expander("Show Tender Document"):
        st.write(st.session_state['tender_doc'])

if st.session_state['email']:
    with st.expander("Show Email for shortlisted vendors"):
        st.write(st.session_state['email'])
    
if st.session_state['evaluated_bids'] is not None:
    with st.expander("Show Top Evaluated Bids"):
        st.dataframe(st.session_state['evaluated_bids'])

if st.session_state['negotiation_strategy']:
    with st.expander("Show Negotiation Strategy"):
        st.write(st.session_state['negotiation_strategy'])

if st.session_state['risk_assessment']:
    with st.expander("Show Risk Assessment Report"):
        st.write(st.session_state['risk_assessment'])

if st.session_state['contract_draft']:
    with st.expander("Show Contract Draft"):
        st.write(st.session_state['contract_draft'])

st.header("Download Final Documents")
if st.session_state['rfp_document']:
    st.download_button("Download RFP Document", st.session_state['rfp_document'], file_name="Request_For_Proposal.txt")
if st.session_state['technical_requirements']:
    st.download_button("Download Technical Requirements", st.session_state['technical_requirements'], file_name="Technical_Requirements.txt")
if st.session_state['risk_assessment']:
    st.download_button("Download Risk Assessment Report", st.session_state['risk_assessment'], file_name="Risk_assessment_report.txt")
if st.session_state['contract_draft']:
    st.download_button("Download Contract Draft", st.session_state['contract_draft'], file_name="Contract_Draft.txt")

# Sidebar Configuration
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown(
    """
    This Agentic AI tool automates TransGlobal Industries' procurement process using advanced **GenAI technologies** (LLMs, LangChain, and Streamlit). 
    It transforms business requirements into technical specifications, generates RFPs, streamlines vendor selection, evaluates bids, and simulates negotiation strategies—reducing manual work, improving accuracy, and accelerating decision-making. 🚀
    """
)

# 📌 Display library versions
st.sidebar.markdown("### 📦 Library Versions")
st.sidebar.markdown(f"🔹 **Streamlit**: {st.__version__}")
st.sidebar.markdown(f"🔹 **LangChain**: {langchain.__version__}")
st.sidebar.markdown(f"🔹 **Pandas**: {pd.__version__}")

# Sidebar section
st.sidebar.markdown("---")
st.sidebar.markdown("**Created by Group 2**")

# Create a DataFrame for the table
groupdata = {
    "Name": [
        "Aniket Singh", "Ankit Mamgai", "Anubhav Verma",
        "Rohit Behara", "Sudhanshoo Badwe", "Akshay Patel"
    ],
    "FT Number": [
        "FT251018", "FT251019", "FT251021",
        "FT251066", "FT251093", "FT252010"
    ]
}

groupdf = pd.DataFrame(groupdata)
# Display the table in the sidebar
st.sidebar.markdown("### 👥 Team Members")
st.sidebar.dataframe(groupdf, hide_index=True)
