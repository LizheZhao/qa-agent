from typing import Set, Optional, List, Union, Dict, get_origin, get_args
from dataclasses import dataclass, field, fields, asdict
import os
import pickle
import csv
import logging

import pandas as pd

from src.benchmark.utils import get_benchmark_dir_path, QueryRuntimeError
from src.utils import get_current_client_and_model_group


logger = logging.getLogger(__name__)


@dataclass
class Rejected:
    id: str
    query: str
    intention: Set[str] = field(default_factory=lambda: {'none'})

    def to_csv_row(self) -> Dict[str, str]:
        row = asdict(self)
        row["intention"] = ", ".join(self.intention)

        return row


def parse_csv_value(field_type, raw_value: str):
    """
    Parser: CSV cell -> python value, based on the dataclass field_type.
    """
    if not raw_value.strip():
        # CSV cell is empty
        origin = get_origin(field_type)

        if origin is Union:
            args = get_args(field_type)

            # Case: Optional[Set[str]]
            if set in args:
                return set()

            # Case: Optional[]
            if type(None) in args:
                return None

        # Case: Set[str]
        if origin is set:
            return set()
    else:
        # CSV cell is non-empty
        origin = get_origin(field_type)

        # Case: Set[str]
        if origin is set:
            return set(raw_value.split(", "))

        if origin is Union:
            args = get_args(field_type)

            # Case: Optional[Set[str]]
            for arg in args:
                if get_origin(arg) is set:
                    return set(raw_value.split(", "))

            # Case: Union[int, str]
            if int in args:
                try:
                    return int(raw_value)   # if int conversion is invalid, return na
                except Exception:
                    return raw_value

        # Case: str
        if field_type is str:
            return raw_value

        # Case: int
        if field_type is int:
            return int(raw_value)   # parsing will fail is int conversion is invalid

    raise ValueError(f"Parsing fails. Required field type: {field_type}, got csv cell: {raw_value}")


def csv_serialize_value(field_type, python_value):
    """
    Serializer: python value -> CSV cell (string), based on the dataclass field_type.
    """
    if python_value is None:
        return ""

    origin = get_origin(field_type)

    if origin is set:
        return ", ".join(python_value)

    if origin is Union:
        return str(python_value)

    return str(python_value)


@dataclass(kw_only=True)
class DataFilter:
    """
    Base class
    """
    id: str
    query: str
    intention: Set[str]
    trend: str
    rank: str
    how_many: Union[str, int]
    period_type: Optional[Set[str]] = None
    time: Optional[Set[str]] = None
    media_channel: Optional[Set[str]] = None
    business_driver: Optional[Set[str]] = None
    start: Optional[Set[int]] = None
    end: Optional[Set[int]] = None
    # core_dimension: Optional[Set[str]] = None
    # core_dimension_composite: Optional[Set[str]] = None
    # custom_level: Optional[Set[str]] = None
    # custom_tagging: Optional[Set[str]] = None

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]):
        parsed_kwargs = {}
        for f in fields(cls):
            field_name = f.name
            field_type = f.type
            raw_value = row.get(field_name, "")  # default to empty string if missing
            parsed_val = parse_csv_value(field_type, raw_value)
            parsed_kwargs[field_name] = parsed_val

        return cls(**parsed_kwargs)

    def to_csv_row(self) -> Dict[str, str]:
        result = {}
        for f in fields(self):
            field_name = f.name
            field_type = f.type
            python_value = getattr(self, field_name)
            result[field_name] = csv_serialize_value(field_type, python_value)
        return result

    @classmethod
    def from_api(
            cls,
            id: str,
            query: str,
            data_filter: Dict,
    ):
        valid_fields = {f.name for f in fields(cls)}

        # Remove irrelevant key-value pairs
        transformed_data_filter = {k: v for k, v in data_filter.items() if k in valid_fields}

        # Convert List to Set
        for key, value in transformed_data_filter.items():
            if isinstance(value, list):
                transformed_data_filter[key] = set(value)
            # if isinstance(value, dict):
            #     transformed_data_filter[key] = set(value.keys())

        transformed_data_filter = {
            "id": id,
            "query": query,
            **transformed_data_filter,
        }

        return cls(**transformed_data_filter)

    @classmethod
    def from_csv_row_empty(cls, row: Dict[str, str]):
        parsed_kwargs = {}
        for f in fields(cls):
            field_name = f.name
            parsed_kwargs[field_name] = None

        parsed_kwargs["id"] = row.get("id")
        parsed_kwargs["query"] = row.get("query")

        return cls(**parsed_kwargs)

    def validate(self):
        ...


@dataclass
class DataFilterColgusTp(DataFilter):
    product: Optional[Set[str]] = None
    product_halo: Optional[Set[str]] = None


@dataclass
class DataFilterColgusAhw(DataFilter):
    product: Optional[Set[str]] = None
    product_focused_media: Optional[Set[str]] = None


@dataclass
class DataFilterColgusIrish(DataFilter):
    category: Optional[Set[str]] = None


@dataclass
class DataFilterFtr(DataFilter):
    sales_channel: Optional[Set[str]] = None
    promo_nonpromo_media: Optional[Set[str]] = None
    online_offline_media: Optional[Set[str]] = None
    base_expansion_sales: Optional[Set[str]] = None


@dataclass
class DataFilterHilsp(DataFilter):
    sales_channel: Set[str]
    brand: Set[str]
    bm_vs_ecomm: Optional[Set[str]] = None
    brand_focused_marketing: Optional[Set[str]] = None
    campaign: Optional[Set[str]] = None
    business_driver_detail: Optional[Set[str]] = None


@dataclass
class DataFilterScotts(DataFilter):
    product: Set[str]
    business_driver_detail: Optional[Set[str]] = None
    retailer_media: Optional[Set[str]] = None
    brand: Optional[Set[str]] = None


@dataclass
class DataFilterLinkedin(DataFilter):
    product: Set[str]
    business_driver_detail: Optional[Set[str]] = None
    retailer_media: Optional[Set[str]] = None
    brand: Optional[Set[str]] = None


data_fitler_map = {
    "COLGUS-3041": DataFilterColgusTp,
    "COLGUS-3042": DataFilterColgusAhw,
    "COLGUS-3043": DataFilter,
    "COLGUS-3044": DataFilterColgusIrish,
    "COLGUS-3045": DataFilter,
    "FTR-1": DataFilterFtr,
    "HILSP-1": DataFilterHilsp,
    "HILSP-3": DataFilterHilsp,
    "SCOTTS-1": DataFilterScotts,
    "LINKEDIN-11": DataFilterLinkedin,
}


def data_filter_factory():
    client_code, model_group_id = get_current_client_and_model_group()

    key = f"{client_code}-{model_group_id}"

    return data_fitler_map[key]


def load_benchmark_csv(gt_file) -> List[Union[DataFilter, Rejected]]:
    df = pd.read_csv(get_benchmark_dir_path() / gt_file, keep_default_na=False)

    data_filter_class = data_filter_factory()
    benchmark_dataset: List[Union[DataFilter, Rejected, QueryRuntimeError]] = []
    run_gt_comp = int(os.getenv("RUN_GT_COMP"))

    logger.info(f"Benchmark dataset contains {len(df)} rows")
    for _, row in df.iterrows():
        if run_gt_comp:
            if row["intention"] == "none":
                response = Rejected(id=row["id"], query=row["query"])
            else:
                response = data_filter_class.from_csv_row(row.to_dict())
        else:
            response = data_filter_class.from_csv_row_empty(row.to_dict())

        benchmark_dataset.append(response)

    return benchmark_dataset


def load_predicted_data_filter_pickle() -> List[Union[DataFilter, QueryRuntimeError]]:
    with open(get_benchmark_dir_path() / f"{os.getenv('RUN_NAME')}_data_filter.pkl", "rb") as f:
        benchmark_dataset = pickle.load(f)

    return benchmark_dataset


def get_data_filter_csv_columns() -> List[str]:
    data_filter_class = data_filter_factory()

    fieldnames = [f.name for f in fields(data_filter_class)]

    for fld in [f.name for f in fields(Rejected)]:
        if fld not in fieldnames:
            fieldnames.append(fld)

    for fld in [f.name for f in fields(QueryRuntimeError)]:
        if fld not in fieldnames:
            fieldnames.append(fld)

    return fieldnames


def export_predicted_data_filter_csv(
        dataset: List[Union[DataFilter, Rejected, QueryRuntimeError]],
):
    """
    Only rows of instance type DataFilter are included.
    Rows of instance type QueryRuntimeError are excluded.
    """
    if not dataset:
        raise RuntimeError("Dataset is empty. Cannot export to CSV.")

    fieldnames = get_data_filter_csv_columns()

    file_path = get_benchmark_dir_path() / f"{os.getenv('RUN_NAME')}_data_filter.csv"
    with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in dataset:
            writer.writerow(item.to_csv_row())

    return None


def find_attribute_differences(
        gt: Union[DataFilter, Rejected],
        pred: Union[DataFilter, QueryRuntimeError],
) -> Dict:
    """
    Return: {"attribute": (instance1.attribute, instance2.attribute)}
    """
    if not isinstance(gt, type(pred)):
        return {"type": (type(gt), type(pred))}

    differences = {}
    for f in fields(gt):
        field_name = f.name
        value1 = getattr(gt, field_name)
        value2 = getattr(pred, field_name)

        if value1 != value2:
            differences[field_name] = (value1, value2)

    return differences


def get_row_eval_metric(
        gt: Union[DataFilter, Rejected],
        pred: Union[DataFilter, QueryRuntimeError],
) -> Dict:
    metrics = {}

    if isinstance(pred, QueryRuntimeError):
        metrics["error"] = 1
        return metrics
    elif not isinstance(pred, type(gt)):
        metrics["error"] = 1
        return metrics
    else:
        metrics["error"] = 0

    comp_fields_gt = [(f.name, f.type) for f in fields(gt) if f.name not in ["id", "query"]]
    comp_fields_pred = [(f.name, f.type) for f in fields(pred) if f.name not in ["id", "query"]]

    if set(comp_fields_gt) != set(comp_fields_pred):
        logger.warning(
            f"pred & gt dataclass fields mismatch:\n{type(gt)}: {comp_fields_gt}\n{type(pred)}: {comp_fields_pred}"
        )

    for field_name, field_type in comp_fields_gt:
        value_gt: Union[Set, str] = getattr(gt, field_name)
        value_pred: Union[Set, str] = getattr(pred, field_name)

        if (get_origin(field_type) is set) or (Set[str] in get_args(field_type)):
            metrics[f"{field_name}_em"] = 1 if value_gt == value_pred else 0

            if value_pred is None or (len(value_pred) == 0):
                metrics[f"{field_name}_precision"] = 1 if value_gt == value_pred else 0
            else:
                value_gt = set() if value_gt is None else value_gt
                metrics[f"{field_name}_precision"] = len(value_gt & value_pred) / len(value_pred)

            if (value_gt is None) or (len(value_gt) == 0):
                metrics[f"{field_name}_recall"] = 1 if value_gt == value_pred else 0
            else:
                value_pred = set() if value_pred is None else value_pred
                metrics[f"{field_name}_recall"] = len(value_gt & value_pred) / len(value_gt)
        else:
            metrics[f"{field_name}"] = 1 if value_gt == value_pred else 0

    return metrics


def convert_list_attribute_to_set_attribute(instances: Union[DataFilter, QueryRuntimeError]) -> None:
    """
    For predicted data_filters, with incorrect type: list
    Convert inplace: from list to set
    """
    for instance in instances:
        if isinstance(instance, DataFilter):
            for f in fields(instance):
                field_value = getattr(instance, f.name)

                if isinstance(field_value, list):
                    setattr(instance, f.name, set(field_value))

    return None


def convert_benchmark_comp_to_csv_row(
        gt: Union[DataFilter, Rejected],
        pred: Union[DataFilter, Rejected],
) -> Dict:
    """
    Return: id, query, error, {field_name}_match, {field_name}_gt, {field_name}_pred
    """

    row = {
        "id": pred.id,
        "query": pred.query,
        "error": ""
    }

    fields_gt = {f.name: f.type for f in fields(gt)}
    fields_pred = {f.name: f.type for f in fields(pred)}

    # merge field names and maintaining order
    ordered_names_gt = [f.name for f in fields(gt)]
    ordered_names_pred = [f.name for f in fields(pred)]
    merged_field_names = list(dict.fromkeys(ordered_names_gt + ordered_names_pred))
    merged_field_names = [x for x in merged_field_names if x not in {"id", "query"}]

    for field_name in merged_field_names:
        field_type_gt = fields_gt.get(field_name)
        field_type_pred = fields_pred.get(field_name)
        value_gt = getattr(gt, field_name, None)
        value_pred = getattr(pred, field_name, None)

        # If pred type and gt type doesn't match, evaluate all _match to 0
        if isinstance(pred, type(gt)):
            row[f"{field_name}_match"] = 1 if value_gt == value_pred else 0
        else:
            row[f"{field_name}_match"] = 0

        # If field_type evaluate to None, then it doesn't exist in Class, ignore / don't serialize
        if field_type_gt is not None:
            row[f"{field_name}_gt"] = csv_serialize_value(field_type_gt, value_gt)
        else:
            row[f"{field_name}_gt"] = ""   # placeholder for maintaining correct column sorting

        if field_type_pred is not None:
            row[f"{field_name}_pred"] = csv_serialize_value(field_type_pred, value_pred)
        else:
            row[f"{field_name}_pred"] = ""

    return row


def export_data_filter_comp_csv(
        dataset_gt: List[Union[DataFilter, Rejected]],
        dataset_pred: List[Union[DataFilter, Rejected, QueryRuntimeError]]
) -> None:
    comps = []
    for gt, pred in zip(dataset_gt, dataset_pred):
        if isinstance(pred, QueryRuntimeError):
            comps.append({
                "id": pred.id,
                "query": pred.query,
                "error": pred.error_message,
                "error_traceback": pred.error_traceback,
            })
        elif isinstance(pred, DataFilter) or isinstance(pred, Rejected):
            comps.append(convert_benchmark_comp_to_csv_row(gt, pred))
        else:
            raise RuntimeError("Unsupported class")

    df_comps = pd.DataFrame(comps)
    df_comps.to_csv(get_benchmark_dir_path() / f"{os.getenv('RUN_NAME')}_data_filter_comp.csv", index=False)

    return None
