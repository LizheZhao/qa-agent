import logging.config
import os

logging.config.fileConfig('logging.conf')
logging.getLogger('src.model.filter_generator').setLevel(logging.DEBUG)
logging.getLogger('src.model.data_filtering').setLevel(logging.DEBUG)

from src.model.filter_generator import generate_ner_filter
from src.model.readout import process_data, response_generate
from src.data.data_interface import ProcessIndicator

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    client_code = os.getenv("CLIENT_CODE")
    model_group_id = int(os.getenv("MODEL_GROUP_ID"))
    query = 'what is roi of gm media tactics'

    process_indicator = ProcessIndicator.from_local(client_code, model_group_id)
    readout_data = generate_ner_filter(client_code, model_group_id, query, process_indicator)
    context_str, data, benchmark_data, benchmark_str, planner_data, pretext, pivot_biz_table, \
        principle_pretext, overall_view, readout_adj = process_data(client_code, model_group_id, readout_data,
                                                                    process_indicator)

    text = response_generate(query, context_str)

    logger.info(text)
