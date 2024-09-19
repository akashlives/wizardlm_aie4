Here's a README for the project based on the provided code and context:

# Evol-AIE4: Evolutionary Question Generation and Answering System

Author: [Akash Shetty](https://www.linkedin.com/in/akash-shetty/)
Acknowledgements: AI Makerspace

Evol-AIE4 is an advanced question generation and answering system that uses evolutionary techniques to create complex, multi-faceted questions from given contexts. This system is designed to enhance the capabilities of Language Models (LLMs) in handling intricate reasoning tasks.

## Features

-   Document loading and embedding generation
-   Evolutionary question generation using various techniques
-   Question criticism and validation
-   Answer generation based on evolved questions
-   Support for multiple evolution techniques, including:
    -   Simple questions
    -   Reasoning questions
    -   Multi-context questions
    -   Conversational questions
    -   Contextual questions
    -   Counterfactual questions
    -   Temporal reasoning questions
    -   Mathematical and quantitative reasoning
    -   Causal chain expansion
    -   Analogical reasoning

## Installation

1. Ensure you have Python 3.12 installed.
2. Clone this repository.
3. Install the required dependencies:

```
poetry install
```

## Usage

The main components of the system are:

1. Document Loader (`agents/document_loader.py`)
2. Evolution Agent (`agents/evolution_agent.py`)
3. Question Critic Agent (`agents/question_critic_agent.py`)
4. Answer Generator (`agents/answer_generator.py`)
5. Question Generator (`agents/question_generator.py`)

To use the system, you typically follow this pipeline:

1. Load documents and generate embeddings
2. Generate initial questions
3. Apply evolution techniques to create complex questions
4. Validate and critique the evolved questions
5. Generate answers for the validated questions

For detailed usage, refer to the individual agent files.

## Configuration

The system uses a `QAState` object to maintain the state throughout the pipeline. You can configure various parameters such as:

-   Maximum number of evolved questions
-   Maximum evolutions per technique
-   Quality threshold for question validation

## Evolution Techniques

The system supports multiple evolution techniques, defined in `agents/evolution_techniques.py`. These techniques are used to generate diverse and complex questions from initial simple questions or contexts.

## Docker Support

Sample file is available, need to be implementedA Dockerfile is provided for containerization. To build the Docker image, run:

```
docker build -t evol-aie4 .
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This project uses various open-source libraries, including LangChain, OpenAI, and others. Please refer to the `pyproject.toml` file for a complete list of dependencies.
