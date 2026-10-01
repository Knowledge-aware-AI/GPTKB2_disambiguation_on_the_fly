import json
from pathlib import Path
import re
from loguru import logger

from prompter_parser.abstract_prompter_parser import AbstractPrompterParser
from prompter_parser.exceptions import ParsingException


def _message_text(body: dict) -> str:
    """Text of the assistant message in a Responses API body.
    Its position in `output` depends on the reasoning effort (a reasoning item may come first), so look it up by type."""
    for item in body.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    return part["text"]
    raise ParsingException("No output_text message in response")


class PromptSchema(AbstractPrompterParser):
    def __init__(
            self,
            elicitation_gpt_model: str,
            disambiguation_gpt_model: str,
            description_gen_gpt_model: str,
    ):

        self.elicitation_gpt_model = elicitation_gpt_model
        self.disambiguation_gpt_model = disambiguation_gpt_model
        self.description_gen_gpt_model = description_gen_gpt_model
        self.prompt_dir = Path(__file__).parent.parent / "prompts"
        self.system_elicitation_prompt = self.load_prompt("elicitation_system.md")
        self.system_ner_prompt = self.load_prompt("ner_system.md")
        self.system_ned1_prompt = self.load_prompt("ned1_separate_system.md")
        self.system_ned2_prompt = self.load_prompt("ned2_separate_system.md")
        self.system_pd_prompt = self.load_prompt("pd_separate_system.md")
        self.system_cd_prompt = self.load_prompt("cd_separate_system.md")
        self.system_nedg_prompt = self.load_prompt("nedg_system.md")
        self.system_pdg_prompt = self.load_prompt("pdg_system.md")
        self.system_cdg_prompt = self.load_prompt("cdg_system.md")

    def load_prompt(self, filename: str) -> str:
        prompt_path = self.prompt_dir / filename
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def get_elicitation_prompt(self, id, label, description) -> dict:
        user_prompt = f'Subject: {label} \nDescription of subject: {description}'
        return {
            "custom_id": id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.elicitation_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_elicitation_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "none"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "name": "triple_array_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "facts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "subject": {"type": "string"},
                                            "predicate": {"type": "string"},
                                            "object": {"type": "string"}
                                        },
                                        "required": ["subject", "predicate", "object"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["facts"],
                            "additionalProperties": False
                        }
                    }
                },
                "max_output_tokens": 8196,
                # "seed": 42,
                "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "json_schema",
                #     "json_schema": {
                #         "name": "triple_array_response",
                #         "strict": True,
                #         "schema": {
                #             "type": "object",
                #             "properties": {
                #                 "facts": {
                #                     "type": "array",
                #                     "items": {
                #                         "type": "object",
                #                         "properties": {
                #                             "subject": {"type": "string"},
                #                             "predicate": {"type": "string"},
                #                             "object": {"type": "string"}
                #                         },
                #                         "required": ["subject", "predicate", "object"],
                #                         "additionalProperties": False
                #                     }
                #                 }
                #             },
                #             "required": ["facts"],
                #             "additionalProperties": False
                #         }
                #     }
                # }
            }
        }

    def get_ner_prompt(self, triple_id, tsubject, tpredicate, tobject) -> dict:
        user_prompt = f"Phrase: {tobject} | Statement: [{tsubject}, {tpredicate}, {tobject}]\n"
        return {
            "custom_id": triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.disambiguation_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_ner_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "minimal"
                },
                "text": {
                    "verbosity": "low",
                    "format":{
                        "type": "json_object"
                        },
                },
                "max_output_tokens": 256,
                # "seed": 42,
                # "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format":{
                #     "type": "json_object"
                #     },
            },
        }
    
    def get_ned1_prompt(self, triple_id, tsubject, tpredicate, tobject, labels, descs) -> dict:

        options_text = "\n\n".join(
            f"{chr(idx+64)}. Entity: \"{label}\" \nDescription: {desc}"
            for idx, (label, desc) in enumerate(zip(labels, descs), 1)
            )
        fixed_options = (
            "F. None of above.\n\n"
            # "G. Unknown entity: I do not recognize the target entity."
            "G. Unsure - the case is ambiguous/there is not enough information to decide.\n"
            )
        final_options_text = f"{options_text}\n\n{fixed_options}"
        user_prompt = (
            f"Target entity: {tobject}\n"
            f"Context triple: [{tsubject}, {tpredicate}, {tobject}]\n"
            f"Options:\n{final_options_text}"
            )
        
        return {
            "custom_id": triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.disambiguation_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_ned1_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "minimal"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "text"
                        } 
                },
                "max_output_tokens": 128,
                # "seed": 42,
                # "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "text"
                #     }
            }
        }
    
    def get_nedg_prompt(self, triple_id, tsubject, tpredicate, tobject) -> dict:
        user_prompt = (
            f"Entity: {tobject}\n"
            f"Triple: [{tsubject}, {tpredicate}, {tobject}]\n"
            )
        
        return {
            "custom_id": triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.description_gen_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_nedg_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "none"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "text"
                        } 
                },
                "max_output_tokens": 128,
                # "seed": 42,
                "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "text"
                #     }
            }
        }
    
    def get_ned2_prompt(self, triple_id, object_label, object_description, labels, descs) -> dict:

        options_text = "\n\n".join(
            f"{chr(idx+64)}. Entity: \"{label}\" \nDescription: {desc}"
            for idx, (label, desc) in enumerate(zip(labels, descs), 1)
            )
        fixed_options = (
            "F. None of above.\n"
            )
        final_options_text = f"{options_text}\n\n{fixed_options}"
        user_prompt = (
            f"Target entity: {object_label}\n"
            f"Target entity description: {object_description}\n"
            f"Options:\n{final_options_text}"
            )
        
        return {
            "custom_id": triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.disambiguation_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_ned2_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "minimal"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "text"
                        } 
                },
                "max_output_tokens": 128,
                # "seed": 42,
                # "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "text"
                #     }
            }
        }
    
    def get_pd_prompt(self, triple_id, tsubject, tpredicate, tobject, labels, descs) -> dict:

        options_text = "\n\n".join(
            f"{chr(idx+64)}. Predicate: \"{label}\" \nDescription: {desc}"
            for idx, (label, desc) in enumerate(zip(labels, descs), 1)
            )
        fixed_options = (
            "F. None of above.\n"
            )
        final_options_text = f"{options_text}\n\n{fixed_options}"
        user_prompt = (
            f"Target predicate: {tpredicate}\n"
            f"Context triple: [{tsubject}, {tpredicate}, {tobject}]\n"
            f"Options:\n{final_options_text}"
            )
        
        return {
            "custom_id": triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.disambiguation_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_pd_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "minimal"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "text"
                        } 
                },
                "max_output_tokens": 128,
                # "seed": 42,
                # "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "text"
                #     }
            }
        }
    
    def get_pdg_prompt(self, triple_id, tpredicate) -> dict:
        user_prompt = (
            f"Predicate: {tpredicate}\n"
            # f"Triple: [{tsubject}, {tpredicate}, {tobject}]\n"
            )
        
        return {
            "custom_id": triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.description_gen_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_pdg_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "none"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "text"
                        } 
                },
                 "max_output_tokens": 128,
                # "seed": 42,
                "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "text"
                #     }
            }
        }
    
    def get_pdg_triple_prompt(self, triple_id, tsubject, tpredicate, tobject) -> dict:
        user_prompt = (
            f"Predicate: {tpredicate}\n"
            f"Triple: [{tsubject}, {tpredicate}, {tobject}]\n"
            )
        
        return {
            "custom_id": triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.description_gen_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_pdg_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "none"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "text"
                        } 
                },
                "max_output_tokens": 128,
                # "seed": 42,
                "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "text"
                #     }
            }
        }

    def get_cd_prompt(self, instance_triple_id, tsubject, tpredicate, tobject, labels, descs) -> dict:

        options_text = "\n\n".join(
            f"{chr(idx+64)}. Class: \"{label}\" \nDescription: {desc}"
            for idx, (label, desc) in enumerate(zip(labels, descs), 1)
            )
        fixed_options = (
            "F. None of above.\n"
            )
        final_options_text = f"{options_text}\n\n{fixed_options}"
        user_prompt = (
            f"Target class: {tobject}\n"
            f"Context triple: [{tsubject}, {tpredicate}, {tobject}]\n"
            f"Options:\n{final_options_text}"
            )
        
        return {
            "custom_id": instance_triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.disambiguation_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_cd_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "minimal"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "text"
                        } 
                },
                "max_output_tokens": 128,
                # "seed": 42,
                # "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "text"
                #     }
            }
        }
    
    def get_cdg_prompt(self, instance_triple_id, tobject) -> dict:
        user_prompt = (
            f"Class: {tobject}\n"
            # f"Triple: [{tsubject}, {tpredicate}, {tobject}]\n"
            )
        
        return {
            "custom_id": instance_triple_id,
            "method": "POST",
            # "url": "/v1/chat/completions",
            "url": "/v1/responses",
            "body": {
                "model": self.description_gen_gpt_model,
                "input": [
                    {
                        "role": "developer",
                        "content": self.system_cdg_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "reasoning": {
                    "effort": "none"
                },
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "text"
                        } 
                },
                "max_output_tokens": 128,
                # "seed": 42,
                "temperature": 0,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "response_format": {
                #     "type": "text"
                #     }
            }
        }

    def parse_elicitation_response(self, response: dict) -> list[dict]:
        # response_object = json.loads(response.strip())

        subject_id = response["custom_id"]
        # choice = response["response"]["body"]["choices"][0]
        body = response["response"]["body"]

        # finish_reason = body["finish_reason"]
        # if finish_reason != "stop":
        #     raise ParsingException(f"finish_reason={finish_reason}")

        # message = choice["message"]
        if "output" not in body:
            raise ParsingException("Invalid response structure: 'output' missing")
        
        # refusal = body["refusal"]
        # if refusal:
        #     raise ParsingException(f"refusal={refusal}")

        output_string = _message_text(body)
        generated_json_object = json.loads(output_string)

        key = "facts"
        if (type(generated_json_object) != dict or key not in generated_json_object):
            raise ParsingException(f"Key '{key}' not found in response")

        raw_triples = []
        for triple in generated_json_object[key]:
            if {"subject", "predicate", "object"}.issubset(triple):
                triple["subject_id"] = subject_id
                raw_triples.append(triple)

        return raw_triples

    def parse_ner_response(self, response: str) -> tuple[str, tuple]:
        response_object = json.loads(response.strip())
        # content = response_object["response"]["body"]["choices"][0]["message"]["content"]
        content = _message_text(response_object['response']['body'])
        generated_json_object = json.loads(content)

        try:
            phrase = generated_json_object["phrase"]
            is_ne = generated_json_object["is_ne"]
            return (phrase, is_ne)
        except Exception as e:
            logger.error(f"Error parsing NER: {e}")
    
    def parse_ned1_response(self, response: str) -> int | str | None:
        parsed_response = json.loads(response.strip())
        # content = parsed_response['response']['body']['choices'][0]['message']['content']
        content = _message_text(parsed_response['response']['body'])

        content = content.strip().upper()
        matches = re.findall(r'\b([A-G])\b', content)
        unique_matches = set(matches)
        if len(unique_matches) == 1:
            letter = list(unique_matches)[0]
            
            if letter == 'F':
                return None
            elif letter == 'G':
                return "unknown"
            else:
                return ord(letter) - 64
        else:
            return None
        
        # content = content.strip()
        # content_lower = content.lower()
        # # for separate ned
        # if "i don't know" in content_lower:
        #     return "unknown"
        # else:
        #     try:
        #         match = re.search(r'\b([1-5])\b', content)
        #         if match:
        #             return int(match.group(1))
        #         else:
        #             return None
        #     except Exception as e:
        #         return None
        # # for combo
        # try:
        #     cleaned_content = re.sub(r'\s+', '', content).replace('.', '')
        #     idx = int(cleaned_content)
        #     if 1 <= idx <= 5:
        #         return idx
        #     else:
        #         return None
        # except Exception as e:
        #     prefix = "None of"
        #     desc_marker = "description: "
        #     try: 
        #         if content.startswith(prefix):
        #             return content.split(desc_marker)[1]
        #     except Exception as e:
        #         return content
    
    def parse_ned2_response(self, response: str) -> int | str | None:
        parsed_response = json.loads(response.strip())
        # content = parsed_response['response']['body']['choices'][0]['message']['content']
        content = _message_text(parsed_response['response']['body'])

        content = content.strip().upper()
        matches = re.findall(r'\b([A-F])\b', content)
        unique_matches = set(matches)
        if len(unique_matches) == 1:
            letter = list(unique_matches)[0]
            
            if letter == 'F':
                return None
            else:
                return ord(letter) - 64
        else:
            return None

    def parse_pd_response(self, response: str) -> int | str | None:
        parsed_response = json.loads(response.strip())
        # content = parsed_response['response']['body']['choices'][0]['message']['content']
        content = _message_text(parsed_response['response']['body'])

        content = content.strip().upper()
        matches = re.findall(r'\b([A-G])\b', content)
        unique_matches = set(matches)
        if len(unique_matches) == 1:
            letter = list(unique_matches)[0]
            
            if letter == 'F':
                return None
            else:
                return ord(letter) - 64
        else:
            return None
        
        # content = content.strip()
        # # for separate pd
        # try:
        #     match = re.search(r'\b([1-5])\b', content)
        #     if match:
        #         return int(match.group(1))
        #     else:
        #         return None
        # except Exception as e:
        #     return None
        # # for combo
        # try:
        #     cleaned_content = re.sub(r'\s+', '', content).replace('.', '')
        #     idx = int(cleaned_content)
        #     if 1 <= idx <= 5:
        #         return idx
        #     else:
        #         return None
        # except Exception as e:
        #     prefix = "None of"
        #     desc_marker = "description: "
        #     try: 
        #         if content.startswith(prefix):
        #             return content.split(desc_marker)[1]
        #     except Exception as e:
        #         return content
    
    def parse_cd_response(self, response: str) -> int | str | None:
        # try:
        parsed_response = json.loads(response.strip())
        # except json.JSONDecodeError:
        #     return None
        # try:
        # content = parsed_response['response']['body']['choices'][0]['message']['content']
        content = _message_text(parsed_response['response']['body'])
        # except (KeyError, IndexError, TypeError):
        #     return None
        # if not isinstance(content, str):
        #     return None

        content = content.strip().upper()
        matches = re.findall(r'\b([A-G])\b', content)
        unique_matches = set(matches)
        if len(unique_matches) == 1:
            letter = list(unique_matches)[0]
            
            if letter == 'F':
                return None
            else:
                return ord(letter) - 64
        else:
            return None

        # content = content.strip()
        # # for separate cd
        # try:
        #     match = re.search(r'\b([1-5])\b', content)
        #     if match:
        #         return int(match.group(1))
        #     else:
        #         return None
        # except Exception as e:
        #     return None
        # # for combo
        # try:
        #     cleaned_content = re.sub(r'\s+', '', content).replace('.', '')
        #     idx = int(cleaned_content)
        #     if 1 <= idx <= 5:
        #         return idx
        #     else:
        #         return None
        # except Exception as e:
        #     prefix = "None of"
        #     desc_marker = "description: "
        #     try: 
        #         if content.startswith(prefix):
        #             return content.split(desc_marker)[1]
        #     except Exception as e:
        #         return content
    
    def parse_dg_response(self, response: str) -> int | str | None:
        parsed_response = json.loads(response.strip())
        # content = parsed_response['response']['body']['choices'][0]['message']['content']
        content = _message_text(parsed_response['response']['body'])
        desc_marker = "Description: "
        if content.startswith(desc_marker):
            return content.split(desc_marker)[1]
        else:
            return content
    
    def count_tokens(self, response: str):
        parsed_response = json.loads(response.strip())
        usage = parsed_response["response"]["body"]["usage"]

        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]

        return prompt_tokens, completion_tokens