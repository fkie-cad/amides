import numpy as np
import pytest

from amides.data import DataBunch
from amides.features.extraction import (
    TokenCountExtractor,
    TfidfExtractor,
    ProcessArgsExtractor,
    CommandlineExtractor,
    LcsDistanceExtractor,
    RatcliffDistanceExtractor,
)
from amides.features.tokenization import (
    Split,
    WhitespaceAsterisk,
    WhitespaceAsteriskSlashMinus,
)

from amides.features.filter import NumericValues, Strings


class TestFilter:
    def test_numeric_values(self):
        token_list = ["0x1234567", "0123", "0x123", "23405000", "abcdefgh", "itbcded"]
        expected = ["0123", "0x123", "abcdefgh", "itbcded"]
        num_values = NumericValues(length=4)

        result = num_values(token_list)
        assert result == expected

    def test_strings(self):
        token_list = ["too_short", "definitely_long_enough_string", "01234"]
        strings = Strings(length=20)

        result = strings(token_list)

        assert result == ["too_short", "01234"]


class TestTokenizer:
    split_cmdlines = [
        (
            '"reg query \\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default\\""',
            [
                '"reg',
                "query",
                '\\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal',
                "Server",
                'Client\\Default\\""',
            ],
        ),
        (
            '"*reg.exe save hklm\\sam C:\\Users\\*\\AppData\\Local\\Temp\\\\~reg_sam.save*"',
            [
                '"*reg.exe',
                "save",
                "hklm\\sam",
                'C:\\Users\\*\\AppData\\Local\\Temp\\\\~reg_sam.save*"',
            ],
        ),
        (
            "*\\rundll32.exe* Shell32.dll,*Control_RunDLL *",
            ["*\\rundll32.exe*", "Shell32.dll,*Control_RunDLL", "*"],
        ),
        (
            "*icacls * /grant Everyone:F /T /C /Q*",
            ["*icacls", "*", "/grant", "Everyone:F", "/T", "/C", "/Q*"],
        ),
        (
            "* /c ping.exe -n 6 127.0.0.1 & type *",
            ["*", "/c", "ping.exe", "-n", "6", "127.0.0.1", "&", "type", "*"],
        ),
    ]

    @pytest.mark.parametrize("string, expected", split_cmdlines)
    def test_split_tokenizer(self, string, expected):
        split = Split()
        result = split(string)

        assert np.array_equal(result, expected)

    ws_ast_cmdlines = [
        (
            '"reg query \\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default\\""',
            [
                '"reg',
                "query",
                '\\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal',
                "Server",
                'Client\\Default\\""',
            ],
        ),
        (
            '"*reg.exe save hklm\\sam C:\\Users\\*\\AppData\\Local\\Temp\\\\~reg_sam.save*"',
            [
                '"',
                "reg.exe",
                "save",
                "hklm\\sam",
                "C:\\Users\\",
                "\\AppData\\Local\\Temp\\\\~reg_sam.save",
                '"',
            ],
        ),
        (
            "*\\rundll32.exe* Shell32.dll,*Control_RunDLL *",
            ["\\rundll32.exe", "Shell32.dll,", "Control_RunDLL"],
        ),
        (
            "*icacls * /grant Everyone:F /T /C /Q*",
            ["icacls", "/grant", "Everyone:F", "/T", "/C", "/Q"],
        ),
        (
            "* /c ping.exe -n 6 127.0.0.1 & type *",
            ["/c", "ping.exe", "-n", "6", "127.0.0.1", "&", "type"],
        ),
    ]

    @pytest.mark.parametrize("string, expected", ws_ast_cmdlines)
    def test_ws_ast_tokenizer(self, string, expected):
        split = WhitespaceAsterisk()
        result = split(string)

        assert np.array_equal(result, expected)

    ws_ast_sla_min_cmdlines = [
        (
            '"reg query \\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default\\""',
            [
                '"reg',
                "query",
                '"HKEY_CURRENT_USER',
                "Software",
                "Microsoft",
                "Terminal",
                "Server",
                "Client",
                "Default",
                '""',
            ],
        ),
        (
            '"*reg.exe save hklm\\sam C:\\Users\\*\\AppData\\Local\\Temp\\\\~reg_sam.save*"',
            [
                '"',
                "reg.exe",
                "save",
                "hklm",
                "sam",
                "C:",
                "Users",
                "AppData",
                "Local",
                "Temp",
                "~reg_sam.save",
                '"',
            ],
        ),
        (
            "*\\rundll32.exe* Shell32.dll,*Control_RunDLL *",
            ["rundll32.exe", "Shell32.dll,", "Control_RunDLL"],
        ),
        (
            "*icacls * /grant Everyone:F /T /C /Q*",
            ["icacls", "grant", "Everyone:F", "T", "C", "Q"],
        ),
        (
            "* /c ping.exe -n 6 127.0.0.1 & type *",
            ["c", "ping.exe", "n", "6", "127.0.0.1", "&", "type"],
        ),
    ]

    @pytest.mark.parametrize("string, expected", ws_ast_sla_min_cmdlines)
    def test_ws_ast_sla_min_tokenizer(self, string, expected):
        split = WhitespaceAsteriskSlashMinus()
        result = split(string)

        assert np.array_equal(result, expected)


class TestProcessArgsExtractor:
    def test_fit_transform(self):
        data = np.array(
            [
                {"process": {"args": "some-process-args"}},
                {"process": {"args": "some-other-args"}},
            ]
        )
        expected = np.array(["some-process-args", "some-other-args"])
        extractor = ProcessArgsExtractor()
        transformed = extractor.fit_transform(data)
        assert np.array_equal(transformed, expected)

    def test_fit_transform_missing_key(self):
        data = np.array([{"process": {"key": "some-process-args"}}])

        extractor = ProcessArgsExtractor()
        process_args = extractor.fit_transform(data)

        assert not process_args

    def test_transform(self):
        data = np.array(
            [
                {"process": {"args": "some-process-args"}},
                {"process": {"args": "some-other-args"}},
            ]
        )
        expected = np.array(["some-process-args", "some-other-args"])
        extractor = ProcessArgsExtractor()
        transformed_data = extractor.transform(data)
        assert np.array_equal(transformed_data, expected)

    def test_transform_missing_key(self):
        data = np.array([{"process": {"key": "some-process-args"}}])

        extractor = ProcessArgsExtractor()
        process_args = extractor.transform(data)
        assert not process_args

    def test_extract(self):
        train_data = DataBunch(
            np.array(
                [
                    {"process": {"args": "some-process-args"}},
                    {"process": {"args": "some-other-args"}},
                ]
            ),
            np.array([0, 1]),
        )
        expected_train_data = DataBunch(
            np.array(["some-process-args", "some-other-args"]), np.array([0, 1])
        )
        test_data = DataBunch(
            np.array([{"process": {"args": "further-process-args"}}]), np.array([1])
        )
        expected_test_data = DataBunch(
            np.array(["further-process-args"]), np.array([1])
        )

        extractor = ProcessArgsExtractor()
        transformed_train_data, transformed_test_data, _ = extractor.extract(
            train_data, test_data
        )

        assert np.array_equal(
            transformed_train_data.samples, expected_train_data.samples
        )
        assert np.array_equal(transformed_test_data.samples, expected_test_data.samples)


class TestCommandlineExtractor:
    def test_fit_transform(self):
        data = np.array(
            [
                {"process": {"command_line": "some-commandline"}},
                {"winlog": {"event_data": {"CommandLine": "some-other-commandline"}}},
            ]
        )
        expected = np.array(["some-commandline", "some-other-commandline"])
        extractor = CommandlineExtractor()
        transformed_data = extractor.fit_transform(data)
        assert np.array_equal(transformed_data, expected)

    def test_fit_transform_missing_key(self):
        data = np.array([{"process": {"key": "some-commandline"}}])

        extractor = CommandlineExtractor()
        transformed_data = extractor.fit_transform(data)
        assert not transformed_data

    def test_transform(self):
        data = np.array(
            [
                {"process": {"command_line": "some-commandline"}},
                {"winlog": {"event_data": {"CommandLine": "some-other-commandline"}}},
            ]
        )
        expected = np.array(["some-commandline", "some-other-commandline"])
        extractor = CommandlineExtractor()
        transformed_data = extractor.transform(data)
        assert np.array_equal(transformed_data, expected)

    def test_transform_missing_key(self):
        data = np.array([{"process": {"key": "some-commandline"}}])

        extractor = CommandlineExtractor()
        transformed_data = extractor.transform(data)
        assert not transformed_data

    def test_extract(self):
        train_data = DataBunch(
            np.array(
                [
                    {"process": {"command_line": "some-commandline"}},
                    {"process": {"command_line": "some-other-commandline"}},
                ]
            ),
            np.array([1, 0]),
        )
        test_data = DataBunch(
            np.array([{"process": {"command_line": "some-further-commandline"}}]),
            np.array([1]),
        )
        expected_train_data = DataBunch(
            np.array(["some-commandline", "some-other-commandline"]), np.array([1, 0])
        )
        expected_test_data = DataBunch(
            np.array(["some-further-commandline"]), np.array([1])
        )

        extractor = CommandlineExtractor()
        transformed_train_data, transformed_test_data, _ = extractor.extract(
            train_data, test_data
        )

        assert np.array_equal(
            transformed_train_data.samples, expected_train_data.samples
        )
        assert np.array_equal(transformed_test_data.samples, expected_test_data.samples)


class TestTokenCountExtractor:
    def test_fit_transform(self):
        data = np.array(
            [
                "These are some sample sequences",
                "In order to test token count vectorization",
                "which is great",
            ]
        )
        expected_shape = (3, 27)
        extractor = TokenCountExtractor()
        transformed_data = extractor.fit_transform(data)
        assert transformed_data.shape == expected_shape

    def test_extract(self):
        train_data = DataBunch(
            np.array(
                [
                    "These are some sample sequences",
                    "In order to test token count vectorization",
                    "which is great",
                ]
            ),
            np.array([0, 1, 1]),
        )
        test_data = DataBunch(
            np.array(["Some test data", "Just for convenience"]), np.array([1, 0])
        )
        expected_train_data_shape = (3, 27)
        expected_test_data_shape = (2, 27)

        extractor = TokenCountExtractor()
        transformed_train_data, transformed_test_data, _ = extractor.extract(
            train_data, test_data
        )

        assert transformed_train_data.shape == expected_train_data_shape
        assert transformed_test_data.shape == expected_test_data_shape


class TestTfidfExtractor:
    def test_fit_tranfsform(self):
        data = np.array(
            [
                "These are some sample sequences",
                "In order to test token count vectorization",
                "which is great",
            ]
        )

        expected_shape = (3, 27)
        extractor = TfidfExtractor()
        transformed_data = extractor.fit_transform(data)
        assert transformed_data.shape == expected_shape

    def test_extract(self):
        train_data = DataBunch(
            np.array(
                [
                    "These are some sample sequences",
                    "In order to test token count vectorization",
                    "which is great",
                ]
            ),
            np.array([0, 1, 1]),
        )
        test_data = DataBunch(
            np.array(["Some test data", "Just for convenience"]), np.array([1, 0])
        )

        expected_train_data_shape = (3, 27)
        expected_test_data_shape = (2, 27)

        extractor = TfidfExtractor()
        transformed_train_data, transformed_test_data, _ = extractor.extract(
            train_data, test_data
        )

        assert transformed_train_data.shape == expected_train_data_shape
        assert transformed_test_data.shape == expected_test_data_shape


class TestLcsDistanceExtractor:
    def test_init(self):
        extractor = LcsDistanceExtractor(["some", "reference", "data"])
        assert extractor

    def test_init_invalid_reference_data(self):
        with pytest.raises(TypeError):
            _ = LcsDistanceExtractor("invalid-reference-data")

    def test_calculate_max_lcs_distances(self):
        data = np.array(["other", "input", "data"])
        extractor = LcsDistanceExtractor(["some", "reference", "data"])
        max_lcs_distances = extractor.calculate_max_lcs_distances(data)
        assert len(max_lcs_distances) == len(data)

    def test_calculate_max_lcs_distances_invalid_sequence_data(self):
        data = "invalid-input-data"
        extractor = LcsDistanceExtractor(["some", "reference", "data"])
        with pytest.raises(TypeError):
            _ = extractor.calculate_max_lcs_distances(data)

    def test_extract(self):
        train_data = DataBunch(np.array(["some", "input", "data"]), np.array([0, 1, 0]))
        test_data = DataBunch(np.array(["other", "input", "data"]), np.array([1, 0, 0]))

        extractor = LcsDistanceExtractor(["some", "reference", "data"])
        _, _, _ = extractor.extract(train_data, test_data)


class TestRatcliffDistanceExtractor:
    def test_init(self):
        extractor = RatcliffDistanceExtractor(["some", "reference", "data"])
        assert extractor

    def test_init_invalid_refernce_data(self):
        with pytest.raises(TypeError):
            _ = RatcliffDistanceExtractor("invalid-reference-data")

    def test_calculate_max_ratcliff_distances(self):
        data = np.array(["other", "input", "data"])
        extractor = RatcliffDistanceExtractor(["some", "reference", "data"])
        max_ratcliff_distances = extractor.calculate_max_ratcliff_distances(data)
        assert len(max_ratcliff_distances) == len(data)

    def test_calculate_max_ratcliff_distances_invalid_sequence_data(self):
        data = "invalid-input-data"
        extractor = RatcliffDistanceExtractor(["some", "reference", "data"])

        with pytest.raises(TypeError):
            _ = extractor.calculate_max_ratcliff_distances(data)

    def test_extract(self):
        train_data = DataBunch(np.array(["some", "input", "data"]), np.array([1, 0, 0]))
        test_data = DataBunch(np.array(["other", "input", "data"]), np.array([0, 1, 0]))
        extractor = RatcliffDistanceExtractor(["some", "reference", "data"])
        _, _, _ = extractor.extract(train_data, test_data)
