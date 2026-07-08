import os
import tqdm
import json
import pandas as pd
import logging.config
logging.config.fileConfig('logging.conf')

from src.benchmark.data_filter import (
    get_benchmark_dir_path
)
from src.utils import load_yaml_file
from src.data.data_interface import ProcessIndicator
from src.model.filter_generator import generate_ner_filter

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    path = "./config/benchmark_mapping.yaml"
    config: dict = load_yaml_file(path)

    for k, v in config.items():
        benchmark_id = k
        client_code = v["CLIENT_CODE"]
        model_group_id = int(v["MODEL_GROUP_ID"])
        benchmark_version = v["VERSION"]
        gt_file = v["GT_FILE"]
        run = v["RUN"]
        run_gt_comp = v["RUN_GT_COMP"]

        os.environ["CLIENT_CODE"] = v["CLIENT_CODE"]
        os.environ["MODEL_GROUP_ID"] = v["MODEL_GROUP_ID"]
        os.environ["BENCHMARK_VERSION"] = v["VERSION"]
        os.environ["RUN_GT_COMP"] = str(v["RUN_GT_COMP"])

        def save_dict_to_jsonl(records: list[dict], filepath: str) -> None:
            with open(filepath, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if run:
            save_path = get_benchmark_dir_path(client_code, model_group_id, benchmark_version)
            data_filter = pd.read_csv(os.path.join(save_path, gt_file))
            process_indicator = ProcessIndicator.from_local(client_code, model_group_id)
            save_file = "ner_output.jsonl"
            for _, row in tqdm.tqdm(data_filter.iterrows(), total=len(data_filter)):
                row_id = row["id"]
                query = row["query"]
                try:
                    readout_data = generate_ner_filter(client_code, model_group_id, query, process_indicator)
                    # save to json
                    res_dict = {"id": row_id,
                                "query": query,
                                "ner_results": readout_data.ner_results,
                                "ner_filter": readout_data.ner_filters
                                }
                    save_dict_to_jsonl([res_dict],
                                       os.path.join(save_path, save_file))
                except Exception as e:
                    logger.error(f"[ERROR: {id}]: {e}")