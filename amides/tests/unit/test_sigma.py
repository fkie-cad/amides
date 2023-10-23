# pylint: disable=missing-docstring
import pytest
import os
import numpy as np

from amides.sigma import (
    extract_field_values_from_filter,
    RuleDataset,
    RuleSetDataset,
    RuleSetDatasetError,
    RuleDatasetError,
)
from amides.events import Events, EventType


def data_path():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "../data"))


def sigma_path():
    return os.path.join(data_path(), "sigma-study")


def benign_pc_events():
    benign_pc_events_path = os.path.join(
        data_path(), "socbed-sample/process_creation/jsonl"
    )
    events = Events(EventType.PROCESS_CREATION, name="process_creation")
    events.load_from_dir(benign_pc_events_path)

    return events


def benign_powershell_events():
    powershell_events_path = os.path.join(data_path(), "socbed-sample/powershell/jsonl")
    events = Events(EventType.POWERSHELL, name="powershell")
    events.load_from_dir(powershell_events_path)

    return events


def pc_events_path():
    return os.path.join(sigma_path(), "events/windows/process_creation")


def pc_rules_path():
    return os.path.join(sigma_path(), "rules/windows/process_creation")


def powershell_events_path():
    return os.path.join(sigma_path(), "events/windows/powershell")


def powershell_rules_path():
    return os.path.join(sigma_path(), "rules/windows/powershell")


def proxy_events_path():
    return os.path.join(sigma_path(), "events/proxyweb")


def proxy_rules_path():
    return os.path.join(sigma_path(), "rules/proxyweb")


class TestMultiFieldVisitor:
    proc_cmdline_filters = [
        ('process.command_line: "*-noni -ep bypass $*"', ["*-noni -ep bypass $*"]),
        (
            "process.command_line: ("
            '"reg query \\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default\\"" OR '
            '"powershell.exe mshta.exe http*" OR '
            '"cmd.exe /c taskkill /im cmd.exe")',
            [
                'reg query \\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default\\"',
                "powershell.exe mshta.exe http*",
                "cmd.exe /c taskkill /im cmd.exe",
            ],
        ),
        (
            '(process.executable: "C:\\Windows\\SysWOW64\\cmd.exe" AND '
            'process.command_line: "*\\Windows\\Caches\\NavShExt.dll *") OR '
            'process.command_line: "*\\AppData\\Roaming\\MICROS\\~1\\Windows\\Caches\\NavShExt.dll,Setting"',
            [
                "*\\Windows\\Caches\\NavShExt.dll *",
                "*\\AppData\\Roaming\\MICROS\\~1\\Windows\\Caches\\NavShExt.dll,Setting",
            ],
        ),
        (
            'process.command_line: ("*reg.exe save hklm\\sam C:\\Users\\*\\AppData\\Local\\Temp\\\\~reg_sam.save*" OR '
            '"*1q2w3e4r@#$@#$@#$*" OR "* -hp1q2w3e4 *" OR "*.dat data03 10000 -p *") OR '
            '(process.command_line|contains: "*process call create*" AND '
            'process.command_line|contains: "* > C:\\Users\\*\\AppData\\Local\\Temp\\\\~*") OR '
            '(process.command_line: "*netstat -aon | find *" AND '
            'process.command_line: "* > C:\\Users\\*\\AppData\\Local\\Temp\\\\~*") OR '
            'process.command_line: "*.255 10 C:\\ProgramData\\\\*"',
            [
                "*reg.exe save hklm\\sam C:\\Users\\*\\AppData\\Local\\Temp\\\\~reg_sam.save*",
                "*1q2w3e4r@#$@#$@#$*",
                "* -hp1q2w3e4 *",
                "*.dat data03 10000 -p *",
                "*process call create*",
                "* > C:\\Users\\*\\AppData\\Local\\Temp\\\\~*",
                "*netstat -aon | find *",
                "* > C:\\Users\\*\\AppData\\Local\\Temp\\\\~*",
                "*.255 10 C:\\ProgramData\\\\*",
            ],
        ),
        (
            '(((process.command_line: "*7z.exe a -v500m -mx9 -r0 -p*" OR '
            '(process.parent.command_line: "*wscript.exe*" AND process.parent.command_line: "*.vbs*" AND '
            'process.command_line: "*rundll32.exe*" AND process.command_line: "*C:\\Windows*" AND '
            'process.command_line: "*.dll,Tk_*")) OR (process.parent.executable: "*\\rundll32.exe" AND '
            'process.parent.command_line|contains: "*C:\\Windows*" AND process.command_line|contains: "*cmd.exe /C *")) OR '
            '(process.command_line: "*rundll32 c:\\windows\\\\\\*" AND process.command_line|contains: "*.dll *")) OR '
            '((process.parent.executable: "*\\rundll32.exe" AND process.executable: "*\\dllhost.exe") AND NOT '
            '(process.command_line: (" " OR "")))',
            [
                "*7z.exe a -v500m -mx9 -r0 -p*",
                "*rundll32.exe*",
                "*C:\\Windows*",
                "*.dll,Tk_*",
                "*cmd.exe /C *",
                "*rundll32 c:\\windows\\\\\\*",
                "*.dll *",
            ],
        ),
        (
            '(process.command_line: "*.cpl" AND NOT (process.command_line: ("*\\System32\\\\*" OR '
            '"*C:\\Windows\\*"))) OR (process.command_line: "*reg add*" AND '
            'process.command_line: "*CurrentVersion\\\\Control Panel\\\\CPLs*")',
            ["*.cpl", "*reg add*", "*CurrentVersion\\\\Control Panel\\\\CPLs*"],
        ),
        (
            'process.executable: ("*\\net.exe" OR "*\\net1.exe") AND process.command_line: "*view*" AND NOT '
            'process.command_line: "*\\\\\\\\*"',
            ["*view*"],
        ),
    ]

    @pytest.mark.parametrize("rule_filter, expected", proc_cmdline_filters)
    def test_extract_proc_cmdline_field_values_from_filter(self, rule_filter, expected):
        extracted_field_values = extract_field_values_from_filter(
            rule_filter, ["process.command_line"]
        )
        assert extracted_field_values == expected

    registry_event_filters = [
        (
            'winlog.event_data.TargetObject: "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\ntkd"',
            [
                "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\ntkd"
            ],
        ),
        (
            'winlog.event_data.IntegrityLevel: "Medium" AND winlog.event_data.TargetObject: "*\\services\\*" '
            'AND winlog.event_data.TargetObject: ("*\\ImagePath" OR "*\\FailureCommand"OR "*\\Parameters\\ServiceDll")',
            [
                "*\\services\\*",
                "*\\ImagePath",
                "*\\FailureCommand",
                "*\\Parameters\\ServiceDll",
            ],
        ),
        (
            'winlog.event_data.TargetObject: "HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Ports*" '
            'AND winlog.event_data.EventType: ("SetValue" OR "DeleteValue" OR "CreateValue") '
            'AND winlog.event_data.Details: ("*.dll*" OR "*.exe*" OR "*.bat*" OR "*.com*" OR "*C:*")',
            [
                "HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Ports*",
                "*.dll*",
                "*.exe*",
                "*.bat*",
                "*.com*",
                "*C:*",
            ],
        ),
        (
            'winlog.event_data.TargetObject: "HKU\\*_Classes\\CLSID\\*\\InProcServer32\\(Default)"'
            'AND NOT winlog.event_data.Details: ("%%systemroot%%\\system32\\*" OR "%%systemroot%%\\SysWow64\\*"'
            'OR "*\\AppData\\Local\\Microsoft\\OneDrive\\*\\FileCoAuthLib64.dll" OR "*\\AppData\\Local\\Microsoft\\OneDrive\\*\\FileSyncShell64.dll"'
            'OR "*\\AppData\\Local\\Microsoft\\TeamsMeetingAddin\\*\\Microsoft.Teams.AddinLoader.dll")',
            ["HKU\\*_Classes\\CLSID\\*\\InProcServer32\\(Default)"],
        ),
    ]

    @pytest.mark.parametrize("rule_filter, expected", registry_event_filters)
    def test_extract_registry_target_object_field_values_from_filter(
        self, rule_filter, expected
    ):
        extracted_field_values = extract_field_values_from_filter(
            rule_filter, ["winlog.event_data.TargetObject", "winlog.event_data.Details"]
        )
        assert extracted_field_values == expected

    powershell_filters = [
        (
            'winlog.event_id: 4104 AND ScriptBlockText: "*Start-Dnscat2*"',
            ["*Start-Dnscat2*"],
        ),
        (
            '(winlog.event_id: 4104 AND (ScriptBlockText|re: "\\$PSHome\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$PSHome\\[" '
            'OR ScriptBlockText|re: "\\$ShellId\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$ShellId\\[" '
            'OR ScriptBlockText|re: "\\$env:Public\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$env:Public\\[" '
            'OR ScriptBlockText|re: "\\$env:ComSpec\\[(\\s*\\d{1,3}\\s*,){2}" '
            'OR ScriptBlockText|re: "\\*mdr\\*\\W\\s*\\)\\.Name" OR ScriptBlockText|re: "\\$VerbosePreference\\.ToString\\(" '
            'OR ScriptBlockText|re: "\\String\\]\\s*\\$VerbosePreference")) OR (winlog.event_id: 4103 '
            'AND (Payload|re: "\\$PSHome\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$PSHome\\[" '
            'OR Payload|re: "\\$ShellId\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$ShellId\\[" '
            'OR Payload|re: "\\$env:Public\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$env:Public\\[" '
            'OR Payload|re: "\\$env:ComSpec\\[(\\s*\\d{1,3}\\s*,){2}" OR Payload|re: "\\*mdr\\*\\W\\s*\\)\\.Name" '
            'OR Payload|re: "\\$VerbosePreference\\.ToString\\(" OR Payload|re: "\\String\\]\\s*\\$VerbosePreference"))',
            [
                "\\$PSHome\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$PSHome\\[",
                "\\$ShellId\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$ShellId\\[",
                "\\$env:Public\\[\\s*\\d{1,3}\\s*\\]\\s*\\+\\s*\\$env:Public\\[",
                "\\$env:ComSpec\\[(\\s*\\d{1,3}\\s*,){2}",
                "\\*mdr\\*\\W\\s*\\)\\.Name",
                "\\$VerbosePreference\\.ToString\\(",
                "\\String\\]\\s*\\$VerbosePreference",
            ],
        ),
        (
            'ScriptBlockText: ("*WMImplant*" OR "* change_user *" OR "* gen_cli *" '
            'OR "* command_exec *" OR "* disable_wdigest *" OR "* disable_winrm *" '
            'OR "* enable_wdigest *" OR "* enable_winrm *" OR "* registry_mod *" '
            'OR "* remote_posh *" OR "* sched_job *" OR "* service_mod *" '
            'OR "* process_kill *" OR "* active_users *" OR "* basic_info *" '
            'OR "* power_off *" OR "* vacant_system *" OR "* logon_events *")',
            [
                "*WMImplant*",
                "* change_user *",
                "* gen_cli *",
                "* command_exec *",
                "* disable_wdigest *",
                "* disable_winrm *",
                "* enable_wdigest *",
                "* enable_winrm *",
                "* registry_mod *",
                "* remote_posh *",
                "* sched_job *",
                "* service_mod *",
                "* process_kill *",
                "* active_users *",
                "* basic_info *",
                "* power_off *",
                "* vacant_system *",
                "* logon_events *",
            ],
        ),
        (
            'Keyless: "del (Get-PSReadlineOption).HistorySavePath" '
            'OR Keyless: "Set-PSReadlineOption\\u2013HistorySaveStyle SaveNothing" '
            'OR Keyless: "Remove-Item (Get-PSReadlineOption).HistorySavePath"\\ \\ '
            'OR Keyless: "rm (Get-PSReadlineOption).HistorySavePath"',
            [
                "del (Get-PSReadlineOption).HistorySavePath",
                "Set-PSReadlineOption\\u2013HistorySaveStyle SaveNothing",
                "Remove-Item (Get-PSReadlineOption).HistorySavePath",
                "rm (Get-PSReadlineOption).HistorySavePath",
            ],
        ),
        ('winlog.event_id: 4104 AND Message: "*New-LocalUser*"', ["*New-LocalUser*"]),
        (
            'Message: ("* -nop -w hidden -c * [Convert]::FromBase64String*" '
            'OR "* -w hidden -noni -nop -c iex(New-Object*" '
            'OR "* -w hidden -ep bypass -Enc*" '
            'OR "*powershell.exe reg add HKCU\\software\\microsoft\\windows\\currentversion\\run*" '
            'OR "*bypass -noprofile -windowstyle hidden (new-object system.net.webclient).download*" '
            'OR "*iex(New-Object Net.WebClient).Download*")',
            [
                "* -nop -w hidden -c * [Convert]::FromBase64String*",
                "* -w hidden -noni -nop -c iex(New-Object*",
                "* -w hidden -ep bypass -Enc*",
                "*powershell.exe reg add HKCU\\software\\microsoft\\windows\\currentversion\\run*",
                "*bypass -noprofile -windowstyle hidden (new-object system.net.webclient).download*",
                "*iex(New-Object Net.WebClient).Download*",
            ],
        ),
    ]

    @pytest.mark.parametrize("rule_filter, expected", powershell_filters)
    def test_extract_powershell_field_values_from_filter(self, rule_filter, expected):
        extracted_field_values = extract_field_values_from_filter(
            rule_filter, ["ScriptBlockText", "Keyless", "Message"]
        )
        assert extracted_field_values == expected

    web_proxy_filter = [
        ('c-uri: "*/asp.asp?ui=*"', ["*/asp.asp?ui=*"]),
        (
            'c-useragent: "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko" '
            'AND cs-host: "www.amazon.com" AND ((cs-method: "GET" '
            'AND c-uri: "/s/ref=nb_sb_noss_1/167-3294888-0262949/field-keywords=books" '
            'AND cs-cookie: "*=csm-hit=s-24KU11BB82RZSYGJ3BDK|1419899012996") '
            'OR (cs-method: "POST" AND c-uri: "/N4215/adj/amzn.us.sr.aps"))',
            [
                "www.amazon.com",
                "/s/ref=nb_sb_noss_1/167-3294888-0262949/field-keywords=books",
                "/N4215/adj/amzn.us.sr.aps",
            ],
        ),
        (
            'c-uri-extension: ("exe" OR "vbs" OR "bat" OR "rar" OR "ps1" OR "doc" OR "docm" '
            'OR "xls" OR "xlsm" OR "pptm" OR "rtf" OR "hta" OR "dll" OR "ws" OR "wsf" OR "sct" '
            'OR "zip")',
            [
                "exe",
                "vbs",
                "bat",
                "rar",
                "ps1",
                "doc",
                "docm",
                "xls",
                "xlsm",
                "pptm",
                "rtf",
                "hta",
                "dll",
                "ws",
                "wsf",
                "sct",
                "zip",
            ],
        ),
        (
            'c-useragent: "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko" '
            'AND cs-uri-query: ("/admin/get.php" OR "/news.php" OR "/login/process.php") '
            'AND cs-method: "POST"',
            ["/admin/get.php", "/news.php", "/login/process.php"],
        ),
        (
            'c-uri-query: ("*/install_flash_player.exe" OR "*/flash_install.php*") '
            'AND NOT c-uri-stem: "*.adobe.com/*"',
            ["*/install_flash_player.exe", "*/flash_install.php*"],
        ),
        (
            'c-useragent: "Microsoft BITS/*" AND NOT r-dns: ("*.com" OR "*.net" OR "*.org")',
            [],
        ),
        (
            'r-dns: "api.telegram.org" AND NOT c-useragent: ("*Telegram*" OR "*Bot*")',
            ["api.telegram.org"],
        ),
        (
            'c-uri-query: ("*/install_flash_player.exe" OR "*/flash_install.php*") AND NOT c-uri-stem: "*.adobe.com/*"',
            ["*/install_flash_player.exe", "*/flash_install.php*"],
        ),
        (
            'c-useragent: "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko" '
            'AND cs-uri-query: ("/admin/get.php" OR "/news.php" OR "/login/process.php") '
            'AND cs-method: "POST"',
            ["/admin/get.php", "/news.php", "/login/process.php"],
        ),
    ]

    @pytest.mark.parametrize("rule_filter, expected", web_proxy_filter)
    def test_extract_web_proxy_field_values_from_filter(self, rule_filter, expected):
        extracted_field_values = extract_field_values_from_filter(
            rule_filter,
            [
                "c-uri",
                "c-uri-extension",
                "c-uri-stem",
                "cs-host",
                "c-uri-query",
                "cs-uri-query",
                "r-dns",
            ],
        )
        assert extracted_field_values == expected


class TestRuleDataset:
    rules_data = [
        (
            os.path.join(sigma_path(), "rules/windows/process_creation/rule_1.yml"),
            os.path.join(sigma_path(), "events/windows/process_creation/rule_1"),
            "rule_1",
            (
                "process.command_line: ("
                '"reg query \\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default\\"" OR '
                '"powershell.exe mshta.exe http*" OR '
                '"cmd.exe /c taskkill /im cmd.exe")'
            ),
        ),
        (
            os.path.join(sigma_path(), "rules/windows/powershell/rule_1.yml"),
            os.path.join(sigma_path(), "events/windows/powershell/rule_1"),
            "rule_1",
            (
                'Keyless: "del (Get-PSReadlineOption).HistorySavePath" OR Keyless: "Set-PSReadlineOption '
                '–HistorySaveStyle SaveNothing" OR Keyless: "Remove-Item (Get-PSReadlineOption).HistorySavePath"'
                ' OR Keyless: "rm (Get-PSReadlineOption).HistorySavePath"'
            ),
        ),
        (
            os.path.join(sigma_path(), "rules/windows/registry/rule_1.yml"),
            os.path.join(sigma_path(), "events/windows/registry/rule_1"),
            "rule_1",
            (
                'winlog.event_data.TargetObject: ("*\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\\\*" '
                'OR "*\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnce\\\\*") AND winlog.event_data.Details: '
                '("*C:\\Windows\\Temp\\\\*" OR "*C:\\$Recycle.bin\\\\*" OR "*C:\\Temp\\\\*" OR "*C:\\Users\\Public\\\\*" '
                'OR "%Public%\\\\*" OR "*C:\\Users\\Default\\\\*" OR "*C:\\Users\\Desktop\\\\*" OR "wscript*" '
                'OR "cscript*")'
            ),
        ),
        (
            os.path.join(sigma_path(), "rules/proxyweb/rule_1.yml"),
            os.path.join(sigma_path(), "events/proxyweb/rule_1"),
            "rule_1",
            'url.full: "*/list/suc?name=*"',
        ),
    ]

    @pytest.mark.parametrize("rule_path,events_path,rule_name,rule_filter", rules_data)
    def test_load_events_and_filter(
        self, rule_path, events_path, rule_name, rule_filter
    ):
        rule_data = RuleDataset()
        rule_data.load_events_and_filter(events_path, rule_path)
        assert rule_data.name == rule_name
        assert rule_data.filter[0] == rule_filter
        assert rule_data.evasions.size > 0

    @pytest.mark.parametrize(
        "rule_path,matches_evasions_path",
        [
            (
                os.path.join(sigma_path(), "rules/windows/process_creation/rule_1.yml"),
                os.path.join(
                    sigma_path(), "events/windows/process_creation/missing_evasions"
                ),
            ),
            (
                os.path.join(sigma_path(), "rules/windows/process_creation/rule_1.yml"),
                os.path.join(
                    sigma_path(), "events/windows/process_creation/missing_matches"
                ),
            ),
        ],
    )
    def test_load_events_and_filter_missing_matches_or_evasions(
        self, rule_path, matches_evasions_path
    ):
        rule_data = RuleDataset()

        rule_data.load_events_and_filter(matches_evasions_path, rule_path)

    def test_load_events_and_filter_missing_properties(self):
        missing_properties = os.path.join(
            sigma_path(), "events/windows/process_creation/missing_properties"
        )
        rule_path = os.path.join(
            sigma_path(), "rules/windows/process_creation/rule_1.yml"
        )
        rule_data = RuleDataset()

        with pytest.raises(RuleDatasetError):
            rule_data.load_events_and_filter(missing_properties, rule_path)

    search_fields_values = [
        (
            os.path.join(sigma_path(), "events/windows/process_creation/rule_1"),
            os.path.join(sigma_path(), "rules/windows/process_creation/rule_1.yml"),
            ["process.command_line"],
            [
                'reg query \\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default\\"',
                "powershell.exe mshta.exe http*",
                "cmd.exe /c taskkill /im cmd.exe",
            ],
        ),
        (
            os.path.join(sigma_path(), "events/windows/powershell/rule_1"),
            os.path.join(sigma_path(), "rules/windows/powershell/rule_1.yml"),
            ["Keyless"],
            [
                "del (Get-PSReadlineOption).HistorySavePath",
                "Set-PSReadlineOption –HistorySaveStyle SaveNothing",
                "Remove-Item (Get-PSReadlineOption).HistorySavePath",
                "rm (Get-PSReadlineOption).HistorySavePath",
            ],
        ),
        (
            os.path.join(sigma_path(), "events/windows/registry/rule_1"),
            os.path.join(sigma_path(), "rules/windows/registry/rule_1.yml"),
            ["winlog.event_data.Details", "winlog.event_data.TargetObject"],
            [
                "*\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\\\*",
                "*\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnce\\\\*",
                "*C:\\Windows\\Temp\\\\*",
                "*C:\\$Recycle.bin\\\\*",
                "*C:\\Temp\\\\*",
                "*C:\\Users\\Public\\\\*",
                "%Public%\\\\*",
                "*C:\\Users\\Default\\\\*",
                "*C:\\Users\\Desktop\\\\*",
                "wscript*",
                "cscript*",
            ],
        ),
        (
            os.path.join(sigma_path(), "events/proxyweb/rule_1"),
            os.path.join(sigma_path(), "rules/proxyweb/rule_1.yml"),
            ["url.full"],
            ["*/list/suc?name=*"],
        ),
    ]

    @pytest.mark.parametrize(
        "matches_evasions_path,rule_path,search_fields,field_values",
        search_fields_values,
    )
    def test_extract_fields_from_filter(
        self, matches_evasions_path, rule_path, search_fields, field_values
    ):
        rule_data = RuleDataset()
        rule_data.load_events_and_filter(matches_evasions_path, rule_path)
        assert (
            rule_data.extract_field_values_from_filter(search_fields=search_fields)
            == field_values
        )

    events_paths = [
        (
            benign_pc_events(),
            os.path.join(sigma_path(), "events/windows/process_creation/rule_1"),
            os.path.join(sigma_path(), "rules/windows/process_creation/rule_1.yml"),
        ),
    ]

    @pytest.mark.parametrize(
        "benign_events,matches_evasions_path,rule_path", events_paths
    )
    def test_create_matches_evasions_train_test_split(
        self,
        benign_events,
        matches_evasions_path,
        rule_path,
    ):
        expected_train_size = expected_test_size = 23
        rule_data = RuleDataset()
        rule_data.load_events_and_filter(matches_evasions_path, rule_path)

        train_test_split = rule_data.create_matches_evasions_train_test_split(
            benign_train_events=benign_events, benign_test_events=benign_events
        )

        train_data = train_test_split.train_data
        assert train_data.size == expected_train_size

        test_data = train_test_split.test_data
        assert test_data.size == expected_test_size

    events_search_fields = [
        (
            benign_pc_events(),
            os.path.join(sigma_path(), "events/windows/process_creation/rule_1"),
            os.path.join(sigma_path(), "rules/windows/process_creation/rule_1.yml"),
            ["process.command_line"],
        ),
        (
            benign_powershell_events(),
            os.path.join(sigma_path(), "events/windows/powershell/rule_1"),
            os.path.join(sigma_path(), "rules/windows/powershell/rule_1.yml"),
            ["Keyless"],
        ),
    ]

    @pytest.mark.parametrize(
        "benign_events,matches_evasions_path,rule_path,search_fields",
        events_search_fields,
    )
    def test_create_filter_evasions_train_test_split(
        self, benign_events, matches_evasions_path, rule_path, search_fields
    ):
        rule_data = RuleDataset()
        rule_data.load_events_and_filter(matches_evasions_path, rule_path)
        train_test_split = rule_data.create_filter_evasions_train_test_split(
            benign_train_events=benign_events,
            benign_test_events=benign_events,
            search_fields=search_fields,
        )

        assert train_test_split.train_data
        assert train_test_split.test_data

    @pytest.mark.parametrize(
        "benign_events,matches_evasions_path,rule_path,search_fields",
        events_search_fields,
    )
    def test_create_filter_evasions_train_test_split_with_seed(
        self, benign_events, matches_evasions_path, rule_path, search_fields
    ):
        rule_data = RuleDataset()
        rule_data.load_events_and_filter(matches_evasions_path, rule_path)

        tt_split = rule_data.create_filter_evasions_train_test_split(
            benign_train_events=benign_events,
            benign_test_events=benign_events,
            search_fields=search_fields,
        )
        other_tt_split = rule_data.create_filter_evasions_train_test_split(
            benign_train_events=benign_events,
            benign_test_events=benign_events,
            search_fields=search_fields,
        )

        assert np.array_equal(
            tt_split.train_data.samples, other_tt_split.train_data.samples
        )
        assert np.array_equal(
            tt_split.train_data.labels, other_tt_split.train_data.labels
        )
        assert np.array_equal(
            tt_split.test_data.samples, other_tt_split.test_data.samples
        )
        assert np.array_equal(
            tt_split.test_data.labels, other_tt_split.test_data.labels
        )

    def test_create_matches_evasions_validation_split(self):
        expected_train_size = 23
        expected_test_size = 21
        expected_valid_size = 22
        events_path = os.path.join(
            sigma_path(), "events/windows/process_creation/rule_1"
        )
        rule_path = os.path.join(
            sigma_path(), "rules/windows/process_creation/rule_1.yml"
        )

        rule_data = RuleDataset()
        rule_data.load_events_and_filter(events_path, rule_path)
        valid_split = rule_data.create_matches_evasions_validation_split(
            benign_train_events=benign_pc_events(),
            benign_test_events=benign_pc_events(),
            benign_valid_events=benign_pc_events(),
            evasions_test_size=0.33,
            evasions_valid_size=0.66,
            evasions_split_seed=42,
        )

        train_data = valid_split.train_data
        assert train_data.size == expected_train_size

        test_data = valid_split.test_data
        assert test_data.size == expected_test_size

        valid_data = valid_split.validation_data
        assert valid_data.size == expected_valid_size

    def test_create_filter_evasions_validaion_split(self):
        expected_train_size = 23
        expected_test_size = 21
        expected_valid_size = 22
        events_path = os.path.join(
            sigma_path(), "events/windows/process_creation/rule_1"
        )
        rule_path = os.path.join(
            sigma_path(), "rules/windows/process_creation/rule_1.yml"
        )

        rule_data = RuleDataset()
        rule_data.load_events_and_filter(events_path, rule_path)
        valid_split = rule_data.create_filter_evasions_validation_split(
            benign_train_events=benign_pc_events(),
            benign_test_events=benign_pc_events(),
            benign_valid_events=benign_pc_events(),
            search_fields=["process.command_line"],
            evasions_test_size=0.33,
            evasions_valid_size=0.66,
            evasions_split_seed=42,
        )

        train_data = valid_split.train_data
        assert train_data.size == expected_train_size

        test_data = valid_split.test_data
        assert test_data.size == expected_test_size

        valid_data = valid_split.validation_data
        assert valid_data.size == expected_valid_size


class TestRuleSetDataset:
    @pytest.fixture
    def empty_data_dir(self, tmpdir):
        data = tmpdir.mkdir("data")
        data.mkdir("events")
        data.mkdir("rules")

        return data

    evasions_rule_paths = [
        (powershell_events_path(), powershell_rules_path()),
        (proxy_events_path(), proxy_rules_path()),
    ]

    @pytest.mark.parametrize("evasions_path,rules_path", evasions_rule_paths)
    def test_load_rule_set_data(self, evasions_path, rules_path):
        rule_set_data = RuleSetDataset()
        rule_set_data.load_rule_set_data(evasions_path, rules_path)
        assert rule_set_data.get_rule_dataset_by_name("rule_1")

    def test_load_rule_set_data_invalid_rules_path(self):
        with pytest.raises(RuleSetDatasetError):
            rule_set_data = RuleSetDataset()
            rule_set_data.load_rule_set_data(pc_events_path(), "/non/existing/path")

    def test_load_rule_set_data_invalid_events_path(self):
        with pytest.raises(RuleSetDatasetError):
            rule_set_data = RuleSetDataset()
            rule_set_data.load_rule_set_data("/non/existing/path", pc_rules_path())

    def test_create_matches_evasions_train_test_split(self):
        expected_train_size = expected_test_size = 26
        benign_events = benign_pc_events()
        rule_set_data = RuleSetDataset()
        rule_set_data.load_rule_set_data(pc_events_path(), pc_rules_path())
        tt_split = rule_set_data.create_matches_evasions_train_test_split(
            benign_train_events=benign_events, benign_test_events=benign_events
        )

        assert tt_split.train_data.size == expected_train_size
        assert tt_split.test_data.size == expected_test_size

    def test_create_matches_evasions_valid_split(self):
        expected_train_size = 26
        expected_test_size = expected_valid_size = 23
        benign_events = benign_pc_events()

        rule_set_data = RuleSetDataset()
        rule_set_data.load_rule_set_data(pc_events_path(), pc_rules_path())
        valid_split = rule_set_data.create_matches_evasions_validation_split(
            benign_train_events=benign_events,
            benign_test_events=benign_events,
            benign_valid_events=benign_events,
            evasions_test_size=0.5,
            evasions_valid_size=0.5,
            evasions_split_seed=42,
        )

        assert valid_split.train_data.size == expected_train_size
        assert valid_split.test_data.size == expected_test_size
        assert valid_split.validation_data.size == expected_valid_size

    rules_data = [
        (
            pc_events_path(),
            pc_rules_path(),
            ["process.command_line"],
            [
                'reg query \\"HKEY_CURRENT_USER\\Software\\Microsoft\\Terminal Server Client\\Default\\"',
                "powershell.exe mshta.exe http*",
                "cmd.exe /c taskkill /im cmd.exe",
                "*.cpl",
                "*reg add*",
                "*CurrentVersion\\\\Control Panel\\\\CPLs*",
            ],
        ),
        (
            powershell_events_path(),
            powershell_rules_path(),
            ["Keyless"],
            [
                "del (Get-PSReadlineOption).HistorySavePath",
                "Set-PSReadlineOption –HistorySaveStyle SaveNothing",
                "Remove-Item (Get-PSReadlineOption).HistorySavePath",
                "rm (Get-PSReadlineOption).HistorySavePath",
            ],
        ),
    ]

    @pytest.mark.parametrize(
        "matches_evasions_path,rules_path,search_fields,field_values", rules_data
    )
    def test_extract_field_values_from_filter(
        self, matches_evasions_path, rules_path, search_fields, field_values
    ):
        rule_set_data = RuleSetDataset()
        rule_set_data.load_rule_set_data(matches_evasions_path, rules_path)
        assert (
            rule_set_data.extract_field_values_from_filter(search_fields=search_fields)
            == field_values
        )

    def test_create_filter_evasions_train_test_split(
        self,
    ):
        expected_train_size = expected_test_size = 26
        benign_events = benign_pc_events()

        rule_set_data = RuleSetDataset()
        rule_set_data.load_rule_set_data(pc_events_path(), pc_rules_path())

        tt_split = rule_set_data.create_filter_evasions_train_test_split(
            benign_train_events=benign_events,
            benign_test_events=benign_events,
            search_fields=["process.command_line"],
        )

        assert tt_split.train_data.size == expected_train_size
        assert tt_split.test_data.size == expected_test_size

    def test_create_filter_evasions_valid_split_with_seed(self):
        benign_events = benign_pc_events()
        rule_set = RuleSetDataset()
        rule_set.load_rule_set_data(pc_events_path(), pc_rules_path())

        valid_split = rule_set.create_filter_evasions_validation_split(
            benign_train_events=benign_events,
            benign_test_events=benign_events,
            benign_valid_events=benign_events,
            search_fields=["process.command_line"],
            evasions_test_size=0.5,
            evasions_valid_size=0.5,
            evasions_split_seed=42,
        )
        other_valid_split = rule_set.create_filter_evasions_validation_split(
            benign_train_events=benign_events,
            benign_test_events=benign_events,
            benign_valid_events=benign_events,
            search_fields=["process.command_line"],
            evasions_test_size=0.5,
            evasions_valid_size=0.5,
            evasions_split_seed=42,
        )

        assert np.array_equal(
            valid_split.train_data.samples, other_valid_split.train_data.samples
        )
        assert np.array_equal(
            valid_split.train_data.labels, other_valid_split.train_data.labels
        )
        assert np.array_equal(
            valid_split.test_data.samples, other_valid_split.test_data.samples
        )
        assert np.array_equal(
            valid_split.test_data.labels, other_valid_split.test_data.labels
        )
        assert np.array_equal(
            valid_split.validation_data.samples,
            other_valid_split.validation_data.samples,
        )
        assert np.array_equal(
            valid_split.validation_data.labels, other_valid_split.validation_data.labels
        )
