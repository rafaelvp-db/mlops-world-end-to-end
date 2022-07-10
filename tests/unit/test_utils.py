"""Unit tests for our Utils module"""

from notebooks import utils


def test_module():
    """Basic sanity checks."""
    
    train_test_split = utils.stratified_split_train_test
    prepare_features = utils.prepare_features
    compute_service_features = utils.compute_service_features
    export_df = utils.export_df
    compute_weights = utils.compute_weights
    write_into_delta_table = utils.write_into_delta_table

    pass