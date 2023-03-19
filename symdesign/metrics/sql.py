from __future__ import annotations

import logging
from time import time

import numpy as np
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from symdesign.resources import sql

# Globals
logger = logging.getLogger(__name__)
# Session = sessionmaker()


# def choose_insert_dialect(session):
#     # Choose insert dialect
#     if session.bind.dialect.name == 'sqlite':
#         insert = sqlite_insert
#     elif session.bind.dialect.name == 'mysql':
#         insert = mysql_insert
#     else:  # if session.bind.dialect.name == 'postgresql':
#         # insert = *_insert
#         raise ValueError(
#             f"{insert_dataframe.__name__} isn't configured for the dialect={session.bind.dialect.name} yet")
#     return insert


def insert_dataframe(session: Session, table: sql.Base, df: pd.DataFrame, mysql: bool = False, **kwargs):
    """Take a formatted pandas DataFrame and insert values into a sqlalchemy session, then commit the transaction

    Args:
        session: A currently open transaction within sqlalchemy
        table: A Class mapped to SQL table with sqlalchemy
        df: The DataFrame with records to insert
        mysql: Whether the database is a MySQL dialect
    """
    if mysql:
        insert = mysql_insert
    else:
        insert = sqlite_insert

    insert_stmt = insert(table)
    # # Get the columns that should be updated
    # new_columns = df.index.names + df.columns.to_list()
    # # logger.debug(f'Provided columns: {new_columns}')
    # excluded_columns = insert_stmt.excluded
    # update_columns = [c for c in excluded_columns if c.name in new_columns]
    # update_dict = {getattr(c, 'name'): c for c in update_columns if not c.primary_key}
    # table_ = table.__table__
    # # Find relevant column indicators to parse the non-primary key non-nullable columns
    # primary_keys = [key for key in table_.primary_key]
    # non_null_keys = [col for col in table_.columns if not col.nullable]
    # index_keys = [key for key in non_null_keys if key not in primary_keys]

    # do_update_stmt = insert_stmt.on_conflict_do_update(
    #     index_elements=index_keys,  # primary_keys,
    #     set_=update_dict
    # )
    # # Can't insert with .returning() until version 2.0...
    # # try:
    # #     result = session.execute(do_update_stmt.returning(table_.id), df.reset_index().to_dict('records'))
    # # except exc.CompileError as error:
    # #     logger.error(error)
    # #     try:
    # #         result = session.execute(insert_stmt.returning(table_.id), df.reset_index().to_dict('records'))
    # #     except exc.CompileError as _error:
    # #         logger.error(_error)
    # # try:
    # This works using insert with conflict, however, doesn't return the auto-incremented ids
    # result = session.execute(do_update_stmt, df.reset_index().to_dict('records'))
    # result = session.execute(insert_stmt, df.reset_index().to_dict('records'))
    start_time = time()
    session.execute(insert_stmt, df.reset_index().to_dict('records'))
    logger.debug(f'Transaction with table "{table.__tablename__}" took {time() - start_time:8f}s')

    # session.commit()
    # # foreign_key = [key.column.name for key in table_.foreign_keys]
    # for key in table_.foreign_keys:
    #     foreign_key_name = key.column.name
    #     table2_ = key.column.table
    #     # Repeat the Find procedure for table2_
    #     # Find relevant column indicators to parse the non-primary key non-nullable columns
    #     primary_keys2 = [key for key in table2_.primary_key]
    #     non_null_keys2 = [col for col in table2_.columns if not col.nullable]
    #     index_keys2 = [key for key in non_null_keys2 if key not in primary_keys2]
    #     # Todo this statement fails due to the error:
    #     #  This backend (sqlite) does not support multiple-table criteria within UPDATE
    #     #  This doesn't appear to be a multiple-table update, but a multiple-table criteria,
    #     #  which is supported by sqlite...
    #     # foreign_key_update_stmt = table.update()\
    #     #     .values({key.parent.name: key.column})\
    #     #     .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys, index_keys2)))
    #     # logger.info(foreign_key_update_stmt)
    #
    #     select_stmt = select(key.column).where(key.parent.is_(None))\
    #         .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys, index_keys2))).scalar_subquery()
    #     foreign_key_update_stmt2 = table.update()\
    #         .values({key.parent.name: select_stmt})
    #     logger.info(foreign_key_update_stmt2)
    #     # session.execute(foreign_key_update_stmt)
    #     start_time = time()
    #     session.execute(foreign_key_update_stmt2)
    #     logger.info(f'Transaction took {time() - start_time:8f}s')

    # session.commit()
    # return result

    # # ORM based method which updates objects with each row .id (Takes much longer time
    # # - https://benchling.engineering/sqlalchemy-batch-inserts-a-module-for-when-youre-inserting-thousands-of-rows-and-it-s-slow-16ece0ef5bf7
    # new_objects = [table(**record) for record in df.reset_index().to_dict('records')]
    # # select(table)
    # session.add_all(new_objects)
    # session.commit()
    # # result = [_object.id for _object in new_objects]
    # return [_object.id for _object in new_objects]

# - DRAFTING SPACE -
# For updating foreign keys, I can use the ORM method, which seems a bit simpler if I refactor properly
# table_instance = table(**record)
# foreign_instance = table1(**other_record)
# table_instance.append(foreign_instance)
# Or using UPDATE table foreign_key_id WHERE table.non_nullable_column == foreign_key_table.non_nullable_column
#  table = table.__table__
#  primary_keys = [key.name for key in table.primary_key]
#  non_null_keys = [col.name for col in table.columns if not col.nullable]
#  index_keys = [key for key in non_null_keys if key not in primary_keys]
# for key in table.foreign_keys:
#  foreign_key_name = key.column.name
#  table2 = key.column.table
#  primary_keys2 = [key.name for key in table2.primary_key]
#  non_null_keys2 = [col.name for col in table2.columns if not col.nullable]
#  index_keys2 = [key for key in non_null_keys2 if key not in primary_keys2]
# table.update()
#      .values({key.parent.name: key.column})
#      .where(*tuple(key1 == key2 for key1, key2 in zip(index_keys1, index_keys2)))


def upsert_dataframe(session: Session, table: sql.Base, df: pd.DataFrame, mysql: bool = False, **kwargs):
    """Take a formatted pandas DataFrame and insert/update values into a sqlalchemy session, then commit the transaction

    Args:
        session: A currently open transaction within sqlalchemy
        table: A Class mapped to SQL table with sqlalchemy
        df: The DataFrame with records to insert
        mysql: Whether the database is a MySQL dialect
    """
    if mysql:
        insert_stmt = mysql_insert(table)
        excluded_columns = insert_stmt.inserted
    else:
        insert_stmt = sqlite_insert(table)
        excluded_columns = insert_stmt.excluded

    # Get the columns that should be updated
    new_columns = df.index.names + df.columns.to_list()
    # logger.debug(f'Provided columns: {new_columns}')
    update_columns = [c for c in excluded_columns if c.name in new_columns]
    update_dict = {c.name: c for c in update_columns if not c.primary_key}
    tablename = table.__tablename__
    if mysql:
        do_update_stmt = insert_stmt.on_duplicate_key_update(
            update_dict
        )
    else:  # SQLite and postgresql are the same
        # Find relevant column indicators to parse the non-primary key non-nullable columns
        unique_constraints = inspect(session.connection()).get_unique_constraints(tablename)
        # Returns
        #  [{'name': '_pose_design_uc', 'column_names': ['pose_id', 'design_id']}]
        table_unique_constraint_keys = set()
        for constraint in unique_constraints:
            table_unique_constraint_keys.update(constraint['column_names'])

        table_ = table.__table__
        unique_constraint_keys = {col.name for col in table_.columns if col.unique}
        index_keys = unique_constraint_keys.union(table_unique_constraint_keys)
        # primary_keys = [key for key in table_.primary_key]
        # non_null_keys = [col for col in table_.columns if not col.nullable]
        # index_keys = [key for key in non_null_keys if key not in primary_keys] \
        #     + unique_constraint_keys
        do_update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=index_keys,  # primary_keys,
            set_=update_dict
        )
    start_time = time()
    # result = session.execute(do_update_stmt, df.reset_index().set_index('id').to_dict('records'))
    session.execute(do_update_stmt, df.reset_index().to_dict('records'))
    logger.debug(f'Transaction with table "{tablename}" took {time() - start_time:8f}s')
    # session.commit()

    # return result.scalars().all()


def format_residues_df_for_write(df: pd.DataFrame) -> pd.DataFrame:
    """Take a typical per-residue DataFrame and orient the top column level (level=0) containing the residue numbers on
    the index innermost level

    Args:
        df: A per-residue DataFrame to transform
    Returns:
        The transformed dataframe
    """
    # df.sort_index(inplace=True)
    df.sort_index(level=0, axis=1, inplace=True, sort_remaining=False)
    # residue_metric_columns = residues.columns.levels[-1].to_list()
    # self.log.debug(f'Residues metrics present: {residue_metric_columns}')

    # Add the pose identifier to the dataframe
    # df = pd.concat([df], keys=[str(self)], axis=0)
    # Place the residue indices from the column names into the index at position -1
    df = df.stack(0)
    # df.index.set_names(['pose', 'design', 'index'], inplace=True)
    df.index.set_names('index', level=-1, inplace=True)

    return df


# def which_dialect(session) -> dict[str, bool]:
#     """Provide the database dialect as a dict with the dialect as the key and the value as True"""
#     return {session.bind.dialect.name: True}


def write_dataframe(session: Session, designs: pd.DataFrame = None, residues: pd.DataFrame = None,
                    poses: pd.DataFrame = None, pose_residues: pd.DataFrame = None,
                    entity_designs: pd.DataFrame = None, update: bool = True, transaction_kwargs: dict = dict()):
    """Format each possible DataFrame type for output via csv or SQL database

    Args:
        session: The active session for which the transaction should proceed
        designs: The typical per-design metric DataFrame where each index is the design id and the columns are
            design metrics
        residues: The typical per-residue metric DataFrame where each index is the design id and the columns are
            (residue index, residue metric)
        poses: The typical per-pose metric DataFrame where each index is the pose id and the columns are
            pose metrics
        pose_residues: The typical per-residue metric DataFrame where each index is the design id and the columns are
            (residue index, residue metric)
        entity_designs: The typical per-design metric DataFrame for Entity instances where each index is the design id
            and the columns are design metrics
        update: Whether the output identifiers are already present in the metrics
        transaction_kwargs: Any keyword arguments that should be passed for the transaction. Automatically populated
            with the database backend as located from the session
    """
    #     job: The resources for the current job
    if update:
        dataframe_function = upsert_dataframe
    else:
        dataframe_function = insert_dataframe

    # If this is the first call, update the dictionary to specify the database dialect
    if transaction_kwargs == dict():
        transaction_kwargs.update({session.bind.dialect.name: True})
        # transaction_kwargs.update(which_dialect(session))
    # else:
    #     input(transaction_kwargs)
    # warn = warned = False
    #
    # def warn_multiple_update_results():
    #     nonlocal warned
    #     if warn and not warned:
    #         logger.warning(
    #             "Performing multiple metrics SQL transactions will only return results for the last transaction")
    #         warned = True

    # job = sym_job.job_resources_factory()
    # engine = job.db.engine
    # session = job.current_session
    if poses is not None:
        # warn = True
        poses.replace({np.nan: None}, inplace=True)
        table = sql.PoseMetrics
        dataframe_function(session, table=table, df=poses, **transaction_kwargs)
        # poses.to_sql(table, con=engine, if_exists='append', index=True)
        # #              dtype=sql.Base.metadata.table[table])
        logger.info(f'Wrote {table.__tablename__} to Database')  # {job.internal_db}')

    if designs is not None:
        # warn_multiple_update_results()
        # warn = True
        designs.replace({np.nan: None}, inplace=True)
        table = sql.DesignMetrics
        dataframe_function(session, table=table, df=designs, **transaction_kwargs)
        # designs.to_sql(table, con=engine, if_exists='append', index=True)
        # #                dtype=sql.Base.metadata.table[table])
        logger.info(f'Wrote {table.__tablename__} to Database')  # {job.internal_db}')

    if entity_designs is not None:
        # warn_multiple_update_results()
        # warn = True
        entity_designs.replace({np.nan: None}, inplace=True)
        table = sql.DesignEntityMetrics
        dataframe_function(session, table=table, df=entity_designs, **transaction_kwargs)
        # designs.to_sql(table, con=engine, if_exists='append', index=True)
        # #                dtype=sql.Base.metadata.table[table])
        logger.info(f'Wrote {table.__tablename__} to Database')  # {job.internal_db}')

    if residues is not None:
        # warn_multiple_update_results()
        # warn = True
        residues = format_residues_df_for_write(residues)
        residues.replace({np.nan: None}, inplace=True)
        table = sql.ResidueMetrics
        dataframe_function(session, table=table, df=residues, **transaction_kwargs)
        # residues.to_sql(table, con=engine, index=True)  # if_exists='append',
        # # Todo Ensure that the 'id' column is present
        # stmt = select(sql.DesignMetrics).where(sql.DesignMetrics.pose.in_(pose_ids))  # text('''SELECT * from residues''')
        # results = session.scalars(stmt)  # execute(stmt)
        logger.info(f'Wrote {table.__tablename__} to Database')  # {job.internal_db}')

    if pose_residues is not None:
        # warn_multiple_update_results()
        # warn = True
        pose_residues = format_residues_df_for_write(pose_residues)
        pose_residues.replace({np.nan: None}, inplace=True)
        table = sql.PoseResidueMetrics
        dataframe_function(session, table=table, df=pose_residues, **transaction_kwargs)
        # residues.to_sql(table, con=engine, index=True)  # if_exists='append',
        # # Todo Ensure that the 'id' column is present
        # stmt = select(sql.DesignMetrics).where(sql.DesignMetrics.pose.in_(pose_ids))  # text('''SELECT * from residues''')
        # results = session.scalars(stmt)  # execute(stmt)
        logger.info(f'Wrote {table.__tablename__} to Database')  # {job.internal_db}')
