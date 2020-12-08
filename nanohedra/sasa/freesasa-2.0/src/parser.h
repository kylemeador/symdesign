/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_FREESASA_YY_PARSER_H_INCLUDED
# define YY_FREESASA_YY_PARSER_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int freesasa_yydebug;
#endif
/* "%code requires" blocks.  */
#line 13 "parser.y" /* yacc.c:1909  */


#ifndef FREESASA_TYPEDEF_YY_SCANNER_T
#define FREESASA_TYPEDEF_YY_SCANNER_T
    typedef void* freesasa_yyscan_t;
#endif


#line 53 "parser.h" /* yacc.c:1909  */

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    T_NUMBER = 258,
    T_ID = 259,
    T_AND = 260,
    T_OR = 261,
    T_NOT = 262,
    T_RESN = 263,
    T_RESI = 264,
    T_SYMBOL = 265,
    T_NAME = 266,
    T_CHAIN = 267,
    ATOM = 268
  };
#endif
/* Tokens.  */
#define T_NUMBER 258
#define T_ID 259
#define T_AND 260
#define T_OR 261
#define T_NOT 262
#define T_RESN 263
#define T_RESI 264
#define T_SYMBOL 265
#define T_NAME 266
#define T_CHAIN 267
#define ATOM 268

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 30 "parser.y" /* yacc.c:1909  */

    const char *value;
    expression *expression;

#line 96 "parser.h" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int freesasa_yyparse (expression **expression, freesasa_yyscan_t scanner);

#endif /* !YY_FREESASA_YY_PARSER_H_INCLUDED  */
