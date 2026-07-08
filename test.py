import os
import logging.config
logging.config.fileConfig('logging.conf')

from src.benchmark.data_filter import (
    get_benchmark_dir_path, load_benchmark_csv, load_predicted_data_filter_pickle
)
from src.benchmark.pipeline import run_benchmark_prediction, run_data_filter_report
from src.utils import load_yaml_file

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

        if run:
            try:
                benchmark_dir_path = get_benchmark_dir_path()
                path_data_filter_pkl = benchmark_dir_path / f"{os.getenv('RUN_NAME')}_data_filter.pkl"
                logger.info(f"Running on test benchmark: {benchmark_dir_path}")
                logger.info(f"Test result will be output to: {benchmark_dir_path}")

                if os.path.exists(path_data_filter_pkl):
                    logger.info("Test result already exist")
                    data_filter_gt = load_benchmark_csv(gt_file)
                    data_filter_pred = load_predicted_data_filter_pickle()
                else:
                    data_filter_gt, data_filter_pred = run_benchmark_prediction(
                        client_code, model_group_id, benchmark_version, gt_file
                    )

                if run_gt_comp:
                    run_data_filter_report(data_filter_gt, data_filter_pred)
            except Exception as e:
                logger.error(e)
