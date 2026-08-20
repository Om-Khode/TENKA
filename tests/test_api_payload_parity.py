"""Close the loop between `runtime.py`'s dataclasses and `payloads.py`'s
wire models.

The two are hand-kept in parallel by construction (payloads.py's module
docstring explains why: camelCasing needs a real model, not a dataclass).
Nothing before this file tied them together -- a field renamed in
`runtime.py` would either vanish from the wire silently (if the payload
model kept its old field name and the route's keyword argument to it just
stopped matching -- caught by `TypeError` at call time, at least) or, worse,
survive as a `None` where the dataclass says it can never be one (caught
only if `Entity.properties` had not already shown what a slipped-through
type/nullability mismatch does to response validation: Finding 1's whole
scope going down as a false 400 instead of describing what changed).

This test enumerates every dataclass in `runtime.py` (plus `AuditEntry`,
which lives in `security.py` but has the same wire-mirroring relationship
with `AuditEntryPayload`) that has a genuine payload counterpart, and checks
two things field-by-field:

  1. Every dataclass field reaches the payload under *some* name -- its own,
     or an explicitly declared rename.
  2. Wherever a field's name matches (after any declared rename), its
     nullability agrees on both sides -- a dataclass field typed `X | None`
     must not map to a payload field typed plainly `X`, and vice versa.

Deliberate differences are declared in `PAIRS` below, not special-cased in
the assertions: a rename (`FileEntry.path` -> `FileEntryPayload.id`), a
dataclass field the wire drops (`TurnRef.accepted`/`.reason`, consumed by the
route to choose 202 vs 409, never serialised), or a payload field sourced
from somewhere other than this dataclass (`CommandRunPayload.command_id`
comes from the route's path parameter, not from `CommandOutcome`). Payload
models that wrap a bare `list[...]`/scalar protocol return rather than
mirroring one dataclass (`PreferencesPayload`, `RootsPayload`, `AbortPayload`,
...) are listed separately in `WRAPPER_PAYLOADS_WITHOUT_A_DATACLASS`, with a
reason, so they are accounted for rather than silently skipped -- the sweep
at the bottom of this file fails if any payload in `payloads.py` is neither
in `PAIRS` nor in that set.
"""
from __future__ import annotations

import dataclasses
import typing

import pytest

from assistant.io.api import payloads, runtime
from assistant.io.api.security import AuditEntry
from assistant.io.api.vault import Device


def _is_nullable(annotation: object) -> bool:
    return type(None) in typing.get_args(annotation)


def _dataclass_nullability(cls: type) -> dict[str, bool]:
    hints = typing.get_type_hints(cls)
    return {f.name: _is_nullable(hints[f.name]) for f in dataclasses.fields(cls)}


def _payload_nullability(cls: type[payloads.CamelModel]) -> dict[str, bool]:
    return {name: _is_nullable(info.annotation) for name, info in cls.model_fields.items()}


@dataclasses.dataclass(frozen=True)
class Pair:
    label: str
    source: type
    payload: type[payloads.CamelModel]
    renames: dict[str, str] = dataclasses.field(default_factory=dict)
    # Dataclass fields the wire never carries (consumed by the route itself
    # to pick a status code or a control-flow branch, never serialised).
    dropped: frozenset[str] = frozenset()
    # Payload fields that exist on the wire but are not sourced from this
    # dataclass at all (a route parameter, a different object entirely).
    extra: frozenset[str] = frozenset()


PAIRS: list[Pair] = [
    Pair("TurnRef -> ChatSendPayload", runtime.TurnRef, payloads.ChatSendPayload,
         dropped=frozenset({"accepted", "reason"})),
    Pair("ChatMessage -> ChatMessagePayload", runtime.ChatMessage, payloads.ChatMessagePayload),
    Pair("ConversationRef -> ConversationRefPayload", runtime.ConversationRef,
         payloads.ConversationRefPayload),
    Pair("ConversationDetail -> ConversationDetailPayload", runtime.ConversationDetail,
         payloads.ConversationDetailPayload),
    Pair("Entity -> EntityPayload", runtime.Entity, payloads.EntityPayload),
    Pair("Fact -> FactPayload", runtime.Fact, payloads.FactPayload),
    Pair("Relationship -> RelationshipPayload", runtime.Relationship, payloads.RelationshipPayload),
    Pair("KnowledgeGraph -> KnowledgeGraphPayload", runtime.KnowledgeGraph,
         payloads.KnowledgeGraphPayload),
    Pair("PreferenceChange -> PreferenceChangePayload", runtime.PreferenceChange,
         payloads.PreferenceChangePayload),
    Pair("PreferenceRecord -> PreferenceRecordPayload", runtime.PreferenceRecord,
         payloads.PreferenceRecordPayload),
    Pair("ProcedureRecord -> ProcedureRecordPayload", runtime.ProcedureRecord,
         payloads.ProcedureRecordPayload),
    Pair("SettingRow -> SettingRowPayload", runtime.SettingRow, payloads.SettingRowPayload),
    Pair("SaveOutcome -> SaveOutcomePayload", runtime.SaveOutcome, payloads.SaveOutcomePayload),
    Pair("PersonalityState -> PersonalityPayload", runtime.PersonalityState, payloads.PersonalityPayload),
    Pair("FileEntry -> FileEntryPayload", runtime.FileEntry, payloads.FileEntryPayload,
         renames={"path": "id"}),
    Pair("FileContent -> FileContentPayload", runtime.FileContent, payloads.FileContentPayload,
         renames={"path": "id", "text": "content"}),
    Pair("CommandDef -> CommandDefPayload", runtime.CommandDef, payloads.CommandDefPayload),
    Pair("CommandOutcome -> CommandRunPayload", runtime.CommandOutcome, payloads.CommandRunPayload,
         dropped=frozenset({"ok"}), extra=frozenset({"command_id"})),
    Pair("StatusInfo -> StatusPayload", runtime.StatusInfo, payloads.StatusPayload),
    Pair("TelemetrySnapshot -> TelemetryPayload", runtime.TelemetrySnapshot, payloads.TelemetryPayload),
    Pair("BackupState -> BackupStatePayload", runtime.BackupState, payloads.BackupStatePayload),
    Pair("EnrolledItem -> EnrolledItemPayload", runtime.EnrolledItem, payloads.EnrolledItemPayload),
    Pair("EnrollmentState -> EnrollmentPayload", runtime.EnrollmentState, payloads.EnrollmentPayload),
    Pair("AuditEntry -> AuditEntryPayload", AuditEntry, payloads.AuditEntryPayload),
    # `Device` lives in vault.py rather than runtime.py -- it is not something
    # the assistant reports, it is something this daemon owns -- but its
    # relationship with the wire is identical, so it belongs in the same
    # check. Nothing is declared as dropped on purpose: the fields a device
    # record holds that must never reach a client (`token_hmac`) are not
    # fields on this dataclass at all, they only exist in `devices.json`.
    #
    # `raises` is declared `extra=`: it is not a fact the vault knows. A live
    # ceiling raise lives in `RaiseStore`, in memory, keyed on the device id
    # and a policy name -- the route joins the two, which is exactly what
    # `extra=` is for.
    Pair("Device -> DevicePayload", Device, payloads.DevicePayload,
         extra=frozenset({"raises"})),
]

# Payload models that wrap a bare `list[...]`, a scalar, or a route
# parameter rather than mirroring one dataclass -- named here, with why, so
# the completeness sweep below has somewhere to point instead of silently
# ignoring them.
WRAPPER_PAYLOADS_WITHOUT_A_DATACLASS: dict[type[payloads.CamelModel], str] = {
    payloads.PreferencesPayload: "wraps list[PreferenceRecord]; the protocol returns the list directly",
    payloads.ProceduresPayload: "wraps list[ProcedureRecord]; the protocol returns the list directly",
    payloads.CommandsPayload: "wraps list[CommandDef]; the protocol returns the list directly",
    payloads.ConversationsPayload: "wraps list[ConversationRef]; the protocol returns the list directly",
    payloads.RootsPayload: "wraps list[str]; the protocol returns the list directly",
    payloads.SettingsPayload: "wraps list[SettingRow]; the protocol returns the list directly",
    payloads.FilesListingPayload: "path echoes the route's own query parameter, "
                                   "entries wraps list[FileEntry]; neither is one dataclass",
    payloads.AuditPayload: "wraps list[AuditEntry]; the route builds the list directly",
    payloads.AbortPayload: "sourced from ChatRuntime.abort()'s bare bool, not a dataclass",
    payloads.RestorePayload: "sourced from SystemRuntime.restore_backup()'s bare bool, not a dataclass",
    payloads.UnlockPayload: "sourced from SystemRuntime.unlock_backup()'s bare bool, not a dataclass",
    payloads.ForgottenPayload: "echoes the route's item_id path parameter, not a dataclass",
    payloads.RemovedPayload: "sourced from MemoryRuntime.forget_all()'s bare int, not a dataclass",
    payloads.DeletedPayload: "echoes the route's own request body path, not a dataclass",
    payloads.ForgetEnrolledPayload: "echoes the route's own path parameters, not a dataclass",
    payloads.DevicesPayload: "wraps list[Device]; the vault returns the list directly",
    payloads.TransportsPayload: "wraps list[TransportPayload]; the route builds the "
                                 "list from the transport registry directly",
    payloads.TransportPayload: "sourced from a ListenerPolicy plus the live TransportSession "
                                "-- `ceiling`, `raisable` and `pairable` are read off "
                                "POLICIES, `running`/`url` off the session, and neither is "
                                "a dataclass this file could pair it with",
    payloads.ListenerPayload: "sourced from the resolved ListenerPolicy for the accepting "
                               "port -- `policy`/`allowBearer` off the policy and `canPair` "
                               "from pairing.py's own refusal predicate, so there is no "
                               "dataclass counterpart to mirror",
    payloads.RevokedPayload: "echoes the route's device_id path parameter, not a dataclass",
    payloads.PairCodePayload: "sourced from the minted PairCode plus the endpoint list "
                              "and the QR the route renders from it; PairCode's own "
                              "`grants`/`label` deliberately never reach the wire",
    payloads.RaisePayload: "sourced from a RaiseGrant plus its store key -- the device "
                            "id and the policy name the record is filed under, neither "
                            "of which is a field on the grant; `granted_by` deliberately "
                            "never reaches the wire, and `expires_at` is a monotonic "
                            "reading converted to seconds remaining at request time",
    payloads.SessionPayload: "sourced from the authenticated Device plus the listener "
                              "policy and the pre-ceiling issued grants stashed on "
                              "request.state by authenticate(), not one dataclass",
}


@pytest.mark.parametrize("pair", PAIRS, ids=[p.label for p in PAIRS])
def test_every_dataclass_field_reaches_the_payload(pair: Pair):
    dc_fields = {f.name for f in dataclasses.fields(pair.source)}
    payload_fields = set(pair.payload.model_fields)

    expected_on_wire = {pair.renames.get(name, name) for name in dc_fields} - pair.dropped
    # dropped fields might collide with a rename target name coincidentally;
    # dropped is keyed by the *dataclass* field name, already excluded above
    # via set difference against the dataclass names before renaming --
    # recompute cleanly instead of relying on order:
    expected_on_wire = {
        pair.renames.get(name, name) for name in dc_fields if name not in pair.dropped
    }

    missing = expected_on_wire - payload_fields
    assert not missing, (
        f"{pair.label}: dataclass field(s) {missing} never reached the payload "
        "-- add the field to the payload model, or declare it in `dropped=` "
        "if the wire deliberately omits it"
    )

    unexplained = payload_fields - expected_on_wire - pair.extra
    assert not unexplained, (
        f"{pair.label}: payload field(s) {unexplained} have no dataclass "
        "source and are not declared in `extra=` -- either they are sourced "
        "from the dataclass under a name this pair doesn't know about "
        "(add a rename) or from somewhere else entirely (declare `extra=`)"
    )


@pytest.mark.parametrize("pair", PAIRS, ids=[p.label for p in PAIRS])
def test_nullability_agrees_on_both_sides(pair: Pair):
    dc_null = _dataclass_nullability(pair.source)
    payload_null = _payload_nullability(pair.payload)

    for dc_name, is_null in dc_null.items():
        if dc_name in pair.dropped:
            continue
        wire_name = pair.renames.get(dc_name, dc_name)
        if wire_name not in payload_null:
            continue  # already failed by the field-reachability test above
        assert payload_null[wire_name] == is_null, (
            f"{pair.label}: {dc_name!r} is "
            f"{'nullable' if is_null else 'non-nullable'} on the dataclass "
            f"but {'nullable' if payload_null[wire_name] else 'non-nullable'} "
            f"on the payload ({wire_name!r}) -- a taught fact's null must "
            "reach the wire as null (Finding 1/hard-won property 1), and a "
            "field the dataclass promises is never null must not be widened "
            "to nullable either"
        )


def test_every_payload_model_is_classified():
    """Sweep payloads.py's own module namespace: every CamelModel subclass
    must be either the payload side of a declared Pair or a declared
    wrapper-without-a-dataclass -- nothing added later falls through
    unclassified.
    """
    paired = {pair.payload for pair in PAIRS}
    wrapped = set(WRAPPER_PAYLOADS_WITHOUT_A_DATACLASS)
    unclassified = []
    for name in dir(payloads):
        obj = getattr(payloads, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, payloads.CamelModel)
            and obj is not payloads.CamelModel
            and obj not in paired
            and obj not in wrapped
        ):
            unclassified.append(name)
    assert not unclassified, (
        f"payload model(s) {unclassified} are neither paired with a "
        "dataclass in PAIRS nor declared in WRAPPER_PAYLOADS_WITHOUT_A_DATACLASS "
        "-- classify them so this file stays the one place field-name and "
        "nullability parity is checked"
    )
