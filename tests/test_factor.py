import pandas as pd
import pytest
from xqdata.dataapi import get_dataapi
from xqdata.mock import MockDataApi

from xqfactor.config import add_api


class TestFactor:
    """
    测试Config功能是否正常
    """

    def setup_method(self):
        """每个测试方法执行前的准备"""
        mockapi: MockDataApi = get_dataapi("mock")
        mockapi.set_mock_info(
            "tradedays",
            {"is_tradeday": "bool"},
            pd.date_range("20240101", "20251231"),
        )
        add_api(mockapi)

    def test_leaffactor(self):
        from xqfactor.config import set_option
        from xqfactor.factor import LeafFactor

        set_option("start_time", "20250101")
        set_option("end_time", "20250630")
        set_option("universe", ["000001.SH", "000002.SH"])
        close = LeafFactor("close")
        assert not close.value.empty

    def test_singleleaf_factor(self):
        from xqfactor.config import set_option
        from xqfactor.factor import SingleLeafFactor

        set_option("start_time", "20250101")
        set_option("end_time", "20250630")
        set_option("universe", ["000001.SH", "000002.SH"])
        benchmark = SingleLeafFactor("benchmark", "000300.SH")
        assert benchmark.value["000001.SH"].equals(benchmark.value["000002.SH"])

    def test_objectedolLeaffactor_factor(self):
        from xqfactor.config import set_option
        from xqfactor.factor import ObjectedLeafFactor

        set_option("start_time", "20250101")
        set_option("end_time", "20250630")
        set_option("universe", ["000001.SH", "000002.SH"])
        index_weight = ObjectedLeafFactor("weight", "000300.SH")
        assert index_weight.value.columns.__len__() == 2

    def test_listedfactor_factor(self):
        from xqfactor.config import set_option
        from xqfactor.factor import ListedFactor

        set_option("start_time", "20250101")
        set_option("end_time", "20250630")
        set_option("universe", ["000001.SH", "000002.SH"])
        listed_info = (
            {
                "code": "000001.SH",
                "listed_date": pd.to_datetime("20240101"),
                "delisted_date": pd.to_datetime("20250201"),
            },
            {
                "code": "000002.SH",
                "listed_date": pd.to_datetime("20250301"),
                "delisted_date": pd.to_datetime("20251231"),
            },
        )
        listed_info_df = pd.DataFrame(listed_info)
        listed = ListedFactor(listed_info_df)
        assert listed.value.columns.__len__() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
