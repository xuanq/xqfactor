import pandas as pd
import pytest
from xqdata.dataapi import get_dataapi
from xqdata.mock import MockDataApi

from xqfactor.config import add_api


class TestConfig:
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

    def test_api(self):
        from xqfactor.config import get_api, global_api, remove_api

        rqapi = get_dataapi("rq")
        add_api(rqapi, "rq")
        assert len(global_api._api) == 2
        remove_api("rq")
        assert len(global_api._api) == 1
        mockapi = get_api()
        # print(mockapi.get_info("tradedays"))
        # assert 1 == 2
        assert isinstance(mockapi, MockDataApi)

    def test_config(self):
        from xqfactor.config import global_config

        global_config.set_option("start_time", "20250101")
        global_config.set_option("end_time", "20250630")
        assert global_config.index().min() < pd.Timestamp("20250101")
        assert global_config.index().max() > pd.Timestamp("20250630")

        from xqfactor.config import Config

        local_config = Config()
        local_config.set_option("start_time", "20250301")
        local_config.set_option("end_time", "20250430")
        assert local_config.index != global_config.index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
