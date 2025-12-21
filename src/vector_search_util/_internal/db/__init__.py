import os, json
import aiosqlite
import sqlite3
from vector_search_util.model import CategoryData, RelationData, TagData, SourceDocumentData, ConditionContainer

# sqlite3
class SQLiteClient:
    initialized: bool = False 
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not SQLiteClient.initialized:
            dirname = os.path.dirname(self.db_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            self.__create_categories_table__()
            self.__create_tags_table__()
            self.__create_relations_table__()
            self.__create_source_documents_table__()
            SQLiteClient.initialized = True
    
    def __create_source_documents_table__(self):
        # DBPropertiesテーブルが存在しない場合は作成する
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    source_id TEXT NOT NULL PRIMARY KEY,
                    source_content TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            conn.commit()

    def __create_categories_table__(self):
        # DBPropertiesテーブルが存在しない場合は作成する
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS categories (
                    name TEXT NOT NULL PRIMARY KEY,
                    description TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            conn.commit()

    # Category間のリレーションを管理するテーブル
    def __create_relations_table__(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    from_node TEXT NOT NULL,
                    to_node TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    metadata TEXT,
                    PRIMARY KEY (from_node, to_node, edge_type),
                    FOREIGN KEY (from_node) REFERENCES categories(name),
                    FOREIGN KEY (to_node) REFERENCES categories(name)
                )
            ''')
            conn.commit()

    def __create_tags_table__(self):
        # DBPropertiesテーブルが存在しない場合は作成する
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    name TEXT NOT NULL,
                    description TEXT,
                    metadata TEXT,
                    PRIMARY KEY (name)
                )
            ''')
            conn.commit()

    def __create_conditions_table__(self):
        # ConditionContainer用のテーブルを作成する
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS conditions (
                    name TEXT NOT NULL PRIMARY KEY,
                    condition_data TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            conn.commit()

    def get_content_by_source_id(self, source_id: str) -> str :
        query = "SELECT source_content FROM documents WHERE source_id = ?"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (source_id,))
            row = cur.fetchone()
            if row:
                return row[0]
            return ""

    # source_documents関連
    async def get_source_documents(self, source_ids: list[str]) -> list[SourceDocumentData]:
        conditions = []
        if source_ids:
            conditions.append("source_id IN ({})".format(",".join("?" * len(source_ids))))
        query = "SELECT source_id, source_content, metadata FROM documents"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(source_ids))
                rows = await cur.fetchall()
                documents = []
                for row in rows:
                    metadata_dict = json.loads(row[2]) if row[2] else {}
                    doc = SourceDocumentData(
                        source_id=row[0],
                        source_content=row[1],
                        metadata=metadata_dict
                    )
                    documents.append(doc)
                return documents

    async def upsert_source_documents(self, documents: list[SourceDocumentData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    INSERT INTO documents (source_id, source_content, metadata)
                    VALUES (?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET 
                        source_content=excluded.source_content,
                        metadata=excluded.metadata
                ''', [
                    (
                        doc.source_id, 
                        doc.source_content, 
                        json.dumps(doc.metadata, ensure_ascii=False) if doc.metadata else None
                    ) 
                    for doc in documents
                ])
            await conn.commit()

    async def delete_source_documents(self, source_ids: list[str]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    DELETE FROM documents WHERE source_id = ?
                ''', [(source_id,) for source_id in source_ids])
            await conn.commit()

    async def delete_all_source_documents(self):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    DELETE FROM documents
                ''')
            await conn.commit()

    async def get_categories(self, names: list[str] = [], conditions: ConditionContainer = ConditionContainer()) -> list[CategoryData]:
        if names:
            conditions.add_in_condition("name", names)


        query = "SELECT name, description, metadata FROM categories"
        conditions_sql = conditions.to_sqlite_sql()
        if conditions_sql:
            query += " WHERE " + conditions_sql

        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                rows = await cur.fetchall()
                categories = [CategoryData(name=row[0], description=row[1], metadata=json.loads(row[2]) if row[2] else {}) for row in rows]
                return categories

    async def delete_categories(self, names: list[str]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    DELETE FROM categories WHERE name = ?
                ''', [(name,) for name in names])
            await conn.commit()

    async def upsert_categories(self, category_list: list[CategoryData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    INSERT INTO categories (name, description, metadata)
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET description=excluded.description, metadata=excluded.metadata
                ''', [(category.name, category.description, json.dumps(category.metadata, ensure_ascii=False) if category.metadata else None) for category in category_list])
            await conn.commit()
    
    async def delete_all_categories(self):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    DELETE FROM categories
                ''')
            await conn.commit()

    async def upsert_new_categories(self, data_list_category_names_set: set[str]):
        existing_category_names_set = set([category.name for category in await self.get_categories()])
        new_category_names_set = data_list_category_names_set - existing_category_names_set
        new_categories = [CategoryData(name=category_name, description="", metadata={}) for category_name in new_category_names_set]
        if new_categories:
            await self.upsert_categories(new_categories)

    # relations関連
    async def get_relations(
            self, 
            from_nodes: list[str] = [], 
            to_nodes: list[str] = [], edge_types: list[str] = [],
            conditions: ConditionContainer = ConditionContainer()
            ) -> list[RelationData]:

        if from_nodes:
            conditions.add_in_condition("from_node", from_nodes)
        if to_nodes:
            conditions.add_in_condition("to_node", to_nodes)
        if edge_types:
            conditions.add_in_condition("edge_type", edge_types)
        query = "SELECT from_node, to_node, edge_type, metadata FROM relations"

        sql_conditions = conditions.to_sqlite_sql()
        if sql_conditions:
            query += " WHERE " + sql_conditions

        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(param for param in from_nodes + to_nodes + edge_types if param))
                rows = await cur.fetchall()
                relations = [RelationData(from_node=row[0], to_node=row[1], edge_type=row[2], metadata=json.loads(row[3]) if row[3] else {}) for row in rows]
                return relations

    async def upsert_relations(self, relations: list[RelationData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    INSERT INTO relations (from_node, to_node, edge_type, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(from_node, to_node, edge_type) DO UPDATE SET metadata=excluded.metadata
                ''', [(relation.from_node, relation.to_node, relation.edge_type, json.dumps(relation.metadata, ensure_ascii=False) if relation.metadata else None) for relation in relations if relation.is_valid()])
            await conn.commit()

    async def delete_relations(self, relations: list[RelationData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    DELETE FROM relations WHERE from_node = ? AND to_node = ? AND edge_type = ?
                ''', [(relation.from_node, relation.to_node, relation.edge_type) for relation in relations])
            await conn.commit()

    async def delete_all_relations(self):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    DELETE FROM relations
                ''')
            await conn.commit()

    async def get_tags(self, names: list[str]) -> list[TagData]:
        conditions = []
        if names:
            conditions.append("name IN ({})".format(",".join("?" * len(names))))
        query = "SELECT name, description, metadata FROM tags"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(names))
                rows = await cur.fetchall()
                tags = [TagData(name=row[0], description=row[1], metadata=json.loads(row[2]) if row[2] else {}) for row in rows]
                return tags
        
    async def upsert_tags(self, tag_list: list[TagData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    INSERT INTO tags (name, description, metadata)
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET description=excluded.description, metadata=excluded.metadata
                ''', [(tag.name, tag.description, json.dumps(tag.metadata, ensure_ascii=False) if tag.metadata else None) for tag in tag_list])
            await conn.commit()

    async def delete_tags(self, names: list[str]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    DELETE FROM tags WHERE name = ?
                ''', [(name,) for name in names])
            await conn.commit()

    async def delete_all_tags(self):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    DELETE FROM tags
                ''')
            await conn.commit()

    async def upsert_new_tags(self, data_list_metadata_keys_set: set[str]):
        existing = {t.name for t in await self.get_tags(list(data_list_metadata_keys_set))}
        new_names = data_list_metadata_keys_set - existing
        new_tags = [TagData(name=n, description="", metadata={}) for n in new_names]
        if new_tags:
            await self.upsert_tags(new_tags)

    async def get_conditions(
        self, name_list: list[str] = [], 
        conditions: ConditionContainer = ConditionContainer()
        ) -> list[ConditionContainer]:
        
        query = "SELECT name, condition_data, metadata FROM conditions"
        if name_list:
            conditions.add_in_condition("name", name_list)
        sql_conditions = conditions.to_sqlite_sql()
        if sql_conditions:
            query += " WHERE " + sql_conditions

        results = []
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(name_list))
                rows = await cur.fetchall()
                for row in rows:
                    condition_data = json.loads(row[1])
                    metadata = json.loads(row[2]) if row[2] else {}
                    condition_container = ConditionContainer(name=row[0], **condition_data, metadata=metadata)
                    results.append(condition_container)
        return results


    async def upsert_conditions(self, conditions: list[ConditionContainer]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    INSERT INTO conditions (name, condition_data, metadata)
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET condition_data=excluded.condition_data, metadata=excluded.metadata
                ''', [(condition.name, json.dumps(condition.model_dump(exclude={"name"}), ensure_ascii=False), json.dumps(condition.metadata, ensure_ascii=False) if condition.metadata else None) for condition in conditions])
            await conn.commit()
    
    async def delete_conditions(self, names: list[str]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    DELETE FROM conditions WHERE name = ?
                ''', [(name,) for name in names])
            await conn.commit()

    async def delete_all_conditions(self):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    DELETE FROM conditions
                ''')
            await conn.commit()
