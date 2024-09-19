from langchain_core.prompts import PromptTemplate
from typing import List, Dict


def create_evolution_prompt(
    name: str, instruction: str, examples: List[Dict], is_initial: bool = False
) -> PromptTemplate:
    """
    Creates a PromptTemplate for evolving questions.

    Args:
        name (str): The name of the evolution technique.
        instruction (str): The instruction for evolving the question.
        examples (List[Dict]): A list of example evolutions.
        is_initial (bool): Whether this is for initial question generation.

    Returns:
        PromptTemplate: The created prompt template.
    """
    if is_initial:
        template = PromptTemplate(
            input_variables=["context", "instruction", "examples"],
            template="""
            You must always output questions as strings. Format the question as a sentence or sentences ending with a question mark without any pleasantries or comments.
            {instruction}

            Examples:
            {examples}

            Context: {context}

            Generated Question:
            """,
        )
        print(f"Created PromptTemplate for {name}: {template}")
        print(f"Input variables: {template.input_variables}")
        return template
    else:
        return PromptTemplate(
            input_variables=["question", "context", "instruction", "examples"],
            template="""
            {instruction}

            Examples:
            {examples}

            Question: {question}
            Context: {context}

            Evolved Question:
            """,
        )


evolution_techniques = [
    (
        "simple_question",
        create_evolution_prompt(
            "simple_question",
            "Generate a simple, straightforward question based on the given context. The question should be easy to answer and focus on a single piece of information.",
            [
                {
                    "context": "The Eiffel Tower, located in Paris, France, was completed in 1889. It stands 324 meters tall and was the tallest man-made structure in the world for 41 years.",
                    "output": "When was the Eiffel Tower completed?",
                },
                {
                    "context": "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. The company's first product was the Apple I personal computer.",
                    "output": "Who were the founders of Apple Inc.?",
                },
                {
                    "context": "The human heart beats an average of 100,000 times per day, pumping about 2,000 gallons of blood through the body.",
                    "output": "How many times does the human heart beat on average per day?",
                },
            ],
            is_initial=True,
        ),
    ),
    (
        "reasoning_question",
        create_evolution_prompt(
            "reasoning_question",
            "Complicate the given question by rewriting it into a multi-hop reasoning question based on the provided context.",
            [
                {
                    "question": "What is the capital of France?",
                    "context": "France is a country in Western Europe. It has several cities, including Paris, Lyon, and Marseille. Paris is not only known for its cultural landmarks like the Eiffel Tower and the Louvre Museum but also as the administrative center.",
                    "output": "Which city, famous for the Eiffel Tower, serves as both a cultural hub and the administrative capital of France?",
                },
                {
                    "question": "How does photosynthesis work?",
                    "context": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. Chlorophyll, a green pigment in plants, plays a crucial role in this process.",
                    "output": "What green pigment in plants is essential for converting light into the chemical energy used in photosynthesis?",
                },
                {
                    "question": "Who invented the telephone?",
                    "context": "Alexander Graham Bell is credited with inventing the telephone in 1876. However, Antonio Meucci had previously developed a device for voice communication over electrical wires in 1849.",
                    "output": "Which inventor's 1849 creation preceded the device patented by Alexander Graham Bell in 1876, both related to voice transmission?",
                },
                {
                    "question": "What is the largest planet in our solar system?",
                    "context": "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than two and a half times that of all the other planets in the solar system combined.",
                    "output": "Which gas giant, with a mass exceeding 2.5 times the combined mass of all other planets, dominates our solar system?",
                },
            ],
        ),
    ),
    (
        "multi_context_question",
        create_evolution_prompt(
            "multi_context_question",
            "Rewrite and complicate the given question in a way that answering it requires information derived from multiple contexts.",
            [
                {
                    "question": "What process turns plants green?",
                    "context": "Chlorophyll is the pigment that gives plants their green color and helps them photosynthesize. Photosynthesis in plants typically occurs in the leaves where chloroplasts are concentrated.",
                    "output": "In which plant structures does the pigment responsible for their verdancy facilitate energy production?",
                },
                {
                    "question": "How does the moon affect Earth's tides?",
                    "context": "The moon's gravitational pull causes the oceans to bulge out in the direction of the moon. Another bulge occurs on the opposite side, since the Earth is also being pulled toward the moon.",
                    "output": "How does the moon's gravitational influence create two distinct tidal bulges on Earth, and what role does Earth's rotation play in this process?",
                },
                {
                    "question": "What is the significance of the Rosetta Stone?",
                    "context": "The Rosetta Stone, discovered in 1799, contains the same text written in three scripts: hieroglyphics, Demotic, and ancient Greek. It was crucial in deciphering Egyptian hieroglyphs.",
                    "output": "How did the trilingual inscription on an artifact discovered in 1799 contribute to unlocking an ancient writing system, and what implications did this have for Egyptology?",
                },
                {
                    "question": "How does a vaccine work?",
                    "context": "Vaccines contain weakened or inactive parts of a particular organism that triggers an immune response within the body. This helps the body recognize and fight the organism in future encounters.",
                    "output": "What components in vaccines stimulate the body's defense mechanisms, and how does this preparation affect the immune system's response to subsequent exposures to the actual pathogen?",
                },
            ],
        ),
    ),
    (
        "conversational_question",
        create_evolution_prompt(
            "conversational_question",
            "Reformat the provided question into a series of follow-up questions for a conversational flow.",
            [
                {
                    "question": "What are the advantages and disadvantages of remote work?",
                    "output": "What are the main benefits of working remotely? And on the flip side, what challenges do remote workers typically face? How does remote work impact team collaboration and communication?",
                },
                {
                    "question": "How does climate change affect biodiversity?",
                    "output": "What are some ways that rising temperatures impact ecosystems? How does this temperature change affect plant and animal species? Can you give examples of species that are particularly vulnerable to climate change?",
                },
                {
                    "question": "What are the key features of blockchain technology?",
                    "output": "Can you explain what a blockchain is in simple terms? How does it ensure security and transparency? What are some real-world applications of blockchain beyond cryptocurrencies?",
                },
                {
                    "question": "How has social media changed the way we communicate?",
                    "output": "In what ways has social media made communication easier? Are there any negative effects on personal relationships? How has it impacted the spread of information and news?",
                },
            ],
        ),
    ),
    (
        "contextual_question",
        create_evolution_prompt(
            "contextual_question",
            "Modify the question to explore different contexts, incorporating information from related documents.",
            [
                {
                    "question": "How does climate change affect polar bears?",
                    "context": "Climate change is causing Arctic sea ice to melt, which polar bears rely on for hunting seals. This affects their food supply and habitat.",
                    "output": "Considering the Arctic ecosystem, how does the melting of sea ice due to climate change impact the hunting behavior and survival of polar bears?",
                },
                {
                    "question": "What are the effects of deforestation?",
                    "context": "Deforestation leads to habitat loss, affects the water cycle, and contributes to climate change by reducing carbon absorption.",
                    "output": "How does the large-scale removal of forests impact local biodiversity, global climate patterns, and the Earth's capacity to regulate atmospheric carbon dioxide?",
                },
                {
                    "question": "How does artificial intelligence impact job markets?",
                    "context": "AI is automating many tasks, creating new job opportunities in tech fields, but also potentially displacing workers in certain industries.",
                    "output": "In the context of evolving technology, how is the integration of AI reshaping employment landscapes across various sectors, and what are the implications for workforce education and skill development?",
                },
                {
                    "question": "What is the importance of biodiversity in ecosystems?",
                    "context": "Biodiversity ensures ecosystem resilience, supports various ecosystem services, and plays a crucial role in maintaining ecological balance.",
                    "output": "Considering the interconnectedness of species in an ecosystem, how does biodiversity contribute to the stability and productivity of natural environments, and what are the potential consequences of its loss on ecosystem services?",
                },
            ],
        ),
    ),
    (
        "counterfactual_question",
        create_evolution_prompt(
            "counterfactual_question",
            "Create a counterfactual version of the question that challenges the original assumptions.",
            [
                {
                    "question": "How did the Industrial Revolution impact urban growth?",
                    "context": "The Industrial Revolution led to rapid urbanization as people moved to cities for factory jobs.",
                    "output": "If the Industrial Revolution had prioritized rural development instead of urban factories, how might population distribution and economic growth have differed?",
                },
                {
                    "question": "What were the effects of the printing press on literacy rates?",
                    "context": "The invention of the printing press in the 15th century led to increased availability of books and higher literacy rates.",
                    "output": "How might the spread of knowledge and literacy rates have evolved if the printing press had not been invented until the 20th century?",
                },
                {
                    "question": "How has the internet changed business communication?",
                    "context": "The internet has revolutionized business communication, enabling instant messaging, video conferencing, and global collaboration.",
                    "output": "If the internet had remained a closed military network, how would modern businesses manage global communication and collaboration?",
                },
                {
                    "question": "What impact did the Green Revolution have on global food production?",
                    "context": "The Green Revolution in the mid-20th century dramatically increased agricultural productivity through the use of new technologies and high-yielding crop varieties.",
                    "output": "How might global food security and agricultural practices have developed if the Green Revolution had focused on traditional farming methods instead of technological innovations?",
                },
            ],
        ),
    ),
    (
        "temporal_reasoning_question",
        create_evolution_prompt(
            "temporal_reasoning_question",
            "Reframe the question to involve temporal reasoning or historical context.",
            [
                {
                    "question": "What are the effects of social media on communication?",
                    "context": "Social media has transformed how people interact, share information, and form communities online.",
                    "output": "How has the evolution of social media platforms over the past two decades altered interpersonal communication patterns compared to pre-internet era interactions?",
                },
                {
                    "question": "How has space exploration advanced?",
                    "context": "Space exploration has progressed from early satellite launches to manned moon landings and now includes plans for Mars missions.",
                    "output": "Trace the trajectory of human space exploration from the first satellite launch in 1957 to current Mars mission plans, highlighting key technological advancements at each stage.",
                },
                {
                    "question": "What is the impact of antibiotics on medicine?",
                    "context": "Antibiotics, first widely used in the 1940s, have revolutionized the treatment of bacterial infections.",
                    "output": "Compare the treatment of bacterial infections before and after the widespread introduction of antibiotics in the 1940s, and discuss how antibiotic resistance has evolved over time.",
                },
                {
                    "question": "How has computer processing power changed?",
                    "context": "Computer processing power has increased exponentially since the first electronic computers in the 1940s.",
                    "output": "Analyze the growth in computer processing power from the 1940s to the present day, explaining how this progression has enabled increasingly complex applications over time.",
                },
            ],
        ),
    ),
    (
        "multi_step_logical_deduction",
        create_evolution_prompt(
            "multi_step_logical_deduction",
            "Create a question that requires multiple logical steps to arrive at the answer, building upon the given context.",
            [
                {
                    "question": "What is the capital of France?",
                    "context": "France is in Europe. All European countries have capitals. Paris is the largest city in France.",
                    "output": "If all European countries have capitals, and France is in Europe with Paris as its largest city, what logical conclusion can we draw about the capital of France?",
                },
                {
                    "question": "How do vaccines work?",
                    "context": "Vaccines contain weakened pathogens. The immune system responds to pathogens by creating antibodies. Antibodies provide future protection against diseases.",
                    "output": "Given that vaccines contain weakened pathogens and the immune system responds to pathogens by creating antibodies, how does this process logically lead to future disease protection?",
                },
            ],
        ),
    ),
    (
        "hypothetical_scenario_generation",
        create_evolution_prompt(
            "hypothetical_scenario_generation",
            "Create a speculative scenario based on the original question to encourage out-of-the-box reasoning.",
            [
                {
                    "question": "How does climate change affect ecosystems?",
                    "context": "Climate change alters temperature and precipitation patterns, affecting plant and animal habitats.",
                    "output": "Imagine a scenario where global temperatures rise by 5°C in the next decade. How might this rapid change impact polar ecosystems, and what cascading effects could we expect in tropical regions?",
                },
                {
                    "question": "What are the effects of social media on society?",
                    "context": "Social media influences communication, information spread, and social interactions.",
                    "output": "In a hypothetical world where all social media platforms suddenly shut down for a year, how might this affect global communication, business operations, and social movements?",
                },
            ],
        ),
    ),
    (
        "mathematical_quantitative_reasoning",
        create_evolution_prompt(
            "mathematical_quantitative_reasoning",
            "Formulate a question that requires mathematical computations and reasoning steps based on the given context.",
            [
                {
                    "question": "How fast do electric cars accelerate?",
                    "context": "Some electric cars can accelerate from 0 to 60 mph in under 3 seconds.",
                    "output": "If an electric car accelerates from 0 to 60 mph in 2.8 seconds, what is its average acceleration in meters per second squared? (Assume 1 mile = 1609 meters)",
                },
                {
                    "question": "What is the impact of compound interest on savings?",
                    "context": "Compound interest allows interest to earn interest over time.",
                    "output": "If you invest $10,000 at an annual interest rate of 5% compounded monthly, how much will your investment be worth after 10 years? Show your calculation steps.",
                },
            ],
        ),
    ),
    (
        "causal_chain_expansion",
        create_evolution_prompt(
            "causal_chain_expansion",
            "Develop a question that explores multiple layers of cause-and-effect relationships based on the original question and context.",
            [
                {
                    "question": "How does deforestation affect climate?",
                    "context": "Deforestation reduces tree cover, which impacts carbon absorption and local weather patterns.",
                    "output": "Explain the causal chain from deforestation to changes in local agriculture: How does reduced tree cover affect soil quality, and how might this impact crop yields and subsequently influence local economies and food security?",
                },
                {
                    "question": "What are the effects of automation on employment?",
                    "context": "Automation can replace certain jobs but also create new ones in different sectors.",
                    "output": "Trace the potential long-term effects of widespread automation: How might it change job markets, impact education systems, alter urban development, and ultimately reshape societal structures and values?",
                },
            ],
        ),
    ),
    (
        "analogical_reasoning",
        create_evolution_prompt(
            "analogical_reasoning",
            "Create a question that requires drawing analogies between different concepts related to the original question.",
            [
                {
                    "question": "How does the human immune system work?",
                    "context": "The immune system defends the body against pathogens using various types of cells and proteins.",
                    "output": "How is the human immune system's response to a new pathogen analogous to a country's defense strategy against an unknown threat? Compare the roles of different immune cells to various defense personnel and systems.",
                },
                {
                    "question": "What is the structure of an atom?",
                    "context": "Atoms consist of a nucleus with protons and neutrons, surrounded by electrons in energy levels.",
                    "output": "Draw an analogy between the structure of an atom and the organization of a solar system. How do the relationships between subatomic particles compare to those between celestial bodies?",
                },
            ],
        ),
    ),
]
