
class AbstractPrompterParser:
    def get_elicitation_prompt(self, id, label, description) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI.
        :param subject_name: The target subject.
        :return: A JSON object.
        """
        raise NotImplementedError

    def get_ner_prompt(self, triple_id, tsubject, tpredicate, tobject) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for ner.
        :param entities: The entities to classify.
        :return: A JSON object.
        """
        raise NotImplementedError
    
    def get_ned1_prompt(self, triple_id, tsubject, tpredicate, tobject, labels, descs) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for ned.
        :param triple_id: id of Triple as context for entity to be disambiguated.
        :return: A JSON object.
        """
        raise NotImplementedError

    def get_ned2_prompt(self, triple_id, object_label, object_description, labels, descs) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for ned.
        :param triple_id: id of Triple as context for entity to be disambiguated.
        :return: A JSON object.
        """
        raise NotImplementedError
    
    def get_pd_prompt(self, triple_id, tsubject, tpredicate, tobject, labels, descs) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for pd.
        :param triple_id: id of Triple as context for predicate to be classified.
        :return: A JSON object.
        """
        raise NotImplementedError
    
    def get_cd_prompt(self, instance_triple_id, tsubject, tpredicate, tobject, labels, descs) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for cd.
        :param instance_triple_id: id of InstanceTriple as context for concept to be classified.
        :return: A JSON object.
        """
        raise NotImplementedError
    
    def get_nedg_prompt(self, triple_id, tsubject, tpredicate, tobject) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for nedg.
        :param triple_id: id of Triple as context for entity description generation.
        :return: A JSON object.
        """
        raise NotImplementedError
    
    def get_pdg_prompt(self, triple_id, tpredicate) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for pdg.
        :param triple_id: id of Triple as context for predicate description generation.
        :return: A JSON object.
        """
        raise NotImplementedError
    
    def get_cdg_prompt(self, instance_triple_id, tobject) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for cdg.
        :param instance_triple_id: id of InstanceTriple as context for concept description generation.
        :return: A JSON object.
        """
        raise NotImplementedError
    
    def get_pdg_triple_prompt(self, triple_id, tsubject, tpredicate, tobject) -> dict:
        """
        Get a JSON object to be used as API request to OpenAI for pdg.
        :param triple_id: id of Triple as context for predicate description generation.
        :return: A JSON object.
        """
        raise NotImplementedError

    def parse_elicitation_response(self, response: dict) -> list[dict]:
        """
        Parse the API response from OpenAI to extract triples. The result is a list of dictionaries, each dictionary
        containing a generated triple, with the keys "subject", "predicate" and "object", and an additional key "subject_name",
        which is the original subject name sent to the API. For example:
        {"subject_name": "Vannevar Bush", "subject": "Vannevar Bush", "predicate": "bornIn", "object": "1890"}
        :param response: The API response.
        :return: A list of dictionaries.
        """
        raise NotImplementedError

    def parse_ner_response(self, response: str) -> tuple[str, tuple]:
        """
        Parse the API response from OpenAI for ner. The result is a tuple where of label (str) and if its ne (boolean). For example:
        ("Vannevar Bush": True)
        :param response: The API response.
        :return: A tuple.
        """
        raise NotImplementedError
    
    def parse_ned1_response(self, response: str) -> int | str | None:
        """
        Parse the API response from OpenAI for ned. The result is a string (when given entity is new to kb) or an integer (when given entity already exists in kb). For example:
        "American electrical engineer and science administrator (1890~1974)" / 1
        :param response: The API response.
        :return: A string or an integer.
        """
        raise NotImplementedError
    
    def parse_ned2_response(self, response: str) -> int | str | None:
        """
        Parse the API response from OpenAI for ned. The result is a string (when given entity is new to kb) or an integer (when given entity already exists in kb). For example:
        "American electrical engineer and science administrator (1890~1974)" / 1
        :param response: The API response.
        :return: A string or an integer.
        """
        raise NotImplementedError
    
    def parse_pd_response(self, response: str) -> int | None:
        """
        Parse the API response from OpenAI for pd. The result is an integer (when given predicate can be mapped to an existing one) or None.
        :param response: The API response.
        :return: A list of dictionaries.
        """
        raise NotImplementedError

    def parse_cd_response(self, response: str) -> int | None:
        """
        Parse the API response from OpenAI for cd. The result is an integer (when given class can be mapped to an existing one) or None.
        :param response: The API response.
        :return: A list of dictionaries.
        """
        raise NotImplementedError
    
    def parse_dg_response(self, response: str) -> int | None:
        """
        Parse the API response from OpenAI for dg. The result is a string.
        :param response: The API response.
        :return: A list of dictionaries.
        """
        raise NotImplementedError
    
    def count_tokens(self, response: str):
        raise NotImplementedError