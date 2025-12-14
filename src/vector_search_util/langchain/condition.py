from abc import ABC, abstractmethod
from typing import Union, Literal, Annotated, Any
from pydantic import BaseModel, Field

class Condition(ABC, BaseModel):
    """Base class for all conditions."""
    @abstractmethod
    def build(self):
        raise NotImplementedError

class EqCondition(Condition):
    field: str = Field(..., description="The field to compare.")
    value: Any = Field(..., description="The value to compare against.")

    def build(self):
        return {self.field: self.value}


class InCondition(Condition):
    field: str = Field(..., description="The field to compare.")
    values: list[Any] = Field(..., description="The list of values to compare against.")

    def build(self):
        return {self.field: {"$in": self.values}}


class ContainsCondition(Condition):
    """MongoDB の部分一致（正規表現）"""
    field: str = Field(..., description="The field to compare.")
    substring: str = Field(..., description="The substring to search for.")

    def build(self):
        return {self.field: {"$regex": self.substring}}


# -------------------------
# 比較条件 ($gte, $lte, $gt, $lt)
# -------------------------

class CompareCondition(Condition):
    field: str = Field(..., description="The field to compare.")
    operator: str = Field(..., description="The comparison operator.")
    value: Any = Field(..., description="The value to compare against.")
    def build(self):
        return {self.field: {self.operator: self.value}}


# -------------------------
# 論理条件 ($and, $or, $not)
# -------------------------

class AndCondition(Condition):
    conditions: list[Condition] = Field(..., description="List of conditions")

    def build(self):
        return {"$and": [c.build() for c in self.conditions]}


class OrCondition(Condition):
    conditions: list[Condition] = Field(..., description="List of conditions")

    def build(self):
        return {"$or": [c.build() for c in self.conditions]}


class NotCondition(Condition):
    condition: Condition = Field(..., description="The condition to negate")

    def build(self):
        # NOT は {field: {"$not": {...}}} の形にする必要がある
        built = self.condition.build()
        field, expr = list(built.items())[0]
        return {field: {"$not": expr}}

# -------------------------
# Query Builder
# -------------------------

class ConditionContainer(BaseModel):
    conditions: list[
        Union[
            EqCondition, InCondition, ContainsCondition, CompareCondition, AndCondition, OrCondition, NotCondition
            ]
        ] = Field(default_factory=list, description="List of conditions")

    # --- 基本条件 ---
    def add_eq_condition(self, field, value):
        self.conditions.append(EqCondition(field=field, value=value))
        return self

    def add_in_condition(self, field, values):
        self.conditions.append(InCondition(field=field, values=values))
        return self

    def add_contains_condition(self, field, substring):
        self.conditions.append(ContainsCondition(field=field, substring=substring))
        return self

    # --- 比較条件 ---
    def add_gte_condition(self, field, value):
        self.conditions.append(CompareCondition(field=field, operator="$gte", value=value))
        return self

    def add_lte_condition(self, field, value):
        self.conditions.append(CompareCondition(field=field, operator="$lte", value=value))
        return self

    def add_gt_condition(self, field, value):
        self.conditions.append(CompareCondition(field=field, operator="$gt", value=value))
        return self

    def add_lt_condition(self, field, value):
        self.conditions.append(CompareCondition(field=field, operator="$lt", value=value))
        return self

    # --- 論理条件 ---
    def add_and_condition(self, conditions):
        self.conditions.append(AndCondition(conditions=conditions))
        return self

    def add_or_condition(self, conditions):
        self.conditions.append(OrCondition(conditions=conditions))
        return self

    def add_not_condition(self, condition):
        self.conditions.append(NotCondition(condition=condition))
        return self

    # --- MongoDB風 dict 生成 ---
    def build(self):
        if len(self.conditions) == 0:
            return {}
        if len(self.conditions) == 1:
            return self.conditions[0].build()
        return {"$and": [c.build() for c in self.conditions]}

    # --- PostgreSQL JSONB SQL 生成 ---
    def to_postgres_sql(self):
        if len(self.conditions) == 0:
            return ""

        translator = PostgresJsonbTranslator()
        return translator.translate(self.build())
    

class PostgresJsonbTranslator:
    def translate(self, condition_dict):
        return self._translate_dict(condition_dict)

    def _translate_dict(self, d):
        clauses = []
        for key, value in d.items():
            if key == "$and":
                sub = [self._translate_dict(v) for v in value]
                clauses.append("(" + " AND ".join(sub) + ")")
            elif key == "$or":
                sub = [self._translate_dict(v) for v in value]
                clauses.append("(" + " OR ".join(sub) + ")")
            else:
                clauses.append(self._translate_field(key, value))
        return " AND ".join(clauses)

    def _translate_field(self, field, expr):
        if isinstance(expr, dict):
            if "$in" in expr:
                vals = ",".join([f"'{v}'" for v in expr["$in"]])
                return f"(cmetadata->>'{field}') IN ({vals})"

            if "$regex" in expr:
                return f"(cmetadata->>'{field}') LIKE '%{expr['$regex']}%'"

            if "$gte" in expr:
                return f"(cmetadata->>'{field}')::numeric >= {expr['$gte']}"

            if "$lte" in expr:
                return f"(cmetadata->>'{field}')::numeric <= {expr['$lte']}"

            if "$gt" in expr:
                return f"(cmetadata->>'{field}')::numeric > {expr['$gt']}"

            if "$lt" in expr:
                return f"(cmetadata->>'{field}')::numeric < {expr['$lt']}"

            if "$not" in expr:
                inner = self._translate_field(field, expr["$not"])
                return f"NOT ({inner})"

        # eq
        return f"(cmetadata->>'{field}') = '{expr}'"