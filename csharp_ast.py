import re


class AST:
    NAME_PATTERN = '\@?[a-zA-Z\_][a-zA-Z0-9\_]*'
    DOTTED_NAME_PATTERN = f'{NAME_PATTERN}(\??\.{NAME_PATTERN})*'
    NUMBER = "([0-9]+(\.[0-9]*)?)(?![0-9\.])d?"
    STRING_PART = '[^"]*'
    STRING = f'@?"{STRING_PART}(\\"{STRING_PART})*"'
    CONSTANT = f"({NUMBER}|{NAME_PATTERN}|{STRING})"

    def __init__(self, children):
        self.children = children

    @classmethod
    def parse(cls, string):
        string = cls.remove_comments(string)
        return __class__(list(map(lambda s: ExprAST.parse(s.strip()),
                                  cls.split(string))))

    @staticmethod
    def remove_comments(string):
        while '//' in string:
            ind = string.index('//')
            if '\n' in string[ind:]:
                ind2 = string.index('\n', ind)
            else:
                ind2 = len(string)
            string = string[:ind] + string[ind2:]
        while "/*" in string:
            ind = string.index('/*')
            ind2 = string.index('*/', ind)
            string = string[:ind] + string[ind2 + 2:]
        return string

    @staticmethod
    def split(string, end='}'):
        opens = '([{'
        closes = ')]}'
        s = 0
        res = [""]
        stg = False
        for i in string:
            res[-1] += i
            if not stg:
                if i in opens:
                    s += 1
                if i in closes:
                    s -= 1
                    if s == 0 and i == end:
                        res.append("")
                if i == '"':
                    stg = True
            else:
                if i == '"':
                    stg = False
        while res and not res[-1].strip():
            res.pop()
        return res

    @staticmethod
    def split_first(string, end='}', opensadd="", closesadd=""):
        opens = '{[(' + opensadd
        closes = '}])' + closesadd
        s = 0
        stg = False
        for i in range(len(string)):
            if not stg:
                if string[i] in opens:
                    s += 1
                elif string[i] in closes:
                    s -= 1
                    if s == 0 and string[i] == end:
                        return string[:i + 1], string[i + 1:]
                elif string[i] == '"':
                    stg = True
            else:
                if string[i] == '"':
                    stg = False
        assert False

    @classmethod
    def get_type(cls, string):
        btp = re.match(cls.DOTTED_NAME_PATTERN, string)
        if not btp:
            return None, None
        btp = btp.group()
        if btp in ['new', 'delegate']:
            return None, None
        l = len(btp)
        if len(string) == l:
            return btp, ""
        if string[l:].strip()[0] == "<":
            return cls.split_first(string, opensadd="<", closesadd=">", end=">")
        if string[l:].strip()[0] == "[":
            start, string = string[:l], string[l:].strip()
            while string and string[0] == "[":
                st, string = cls.split_first(string, end="]")
                start += st
            return start, string
        return btp, string[len(btp):].strip()
    
    @classmethod
    def get_dotted_member_name(cls, string):
        obj = re.match(cls.NAME_PATTERN, string)
        if obj is None:
            return None, None
        ind = obj.end()
        name, string = DottedName(string[:ind]), string[ind:].strip()
        if string and string[0] in ".[":
            dot = string[0] == '.'
            if dot:
                string = string[1:]
                ind = re.match(cls.NAME_PATTERN, string).end()
                name2, string = string[:ind], string[ind:].strip()
                name = DottedName([name, DottedName(name2)])
            else:
                expr, string = cls.split_first(string, end=']')
                expr = ExpressionAST.parse(expr[1:-1].strip())
                string = string.strip()
                name = DottedName([name, expr])
            if string and string[0] in '.[':
                rem, string = cls.get_dotted_member_name("A" + string)
                string = string.strip()
                assert not string or string[0] not in '.['
                name = DottedName(name.children + rem.children[1:])
        return name, string
    
    @staticmethod
    def startswith(string, start):
        return re.match(start + '(?![a-zA-Z0-9\_\@])', string) is not None

    def __repr__(self, tab=0, **kwargs):
        return ('  ' * tab + type(self).__name__ + '\n'
                + '\n'.join([child.__repr__(tab + 1)
                             for child in self.children
                             if child]))


class ExprAST(AST):
    def __init__(self, children):
        super().__init__(children)
        self.definition, self.body = children

    @classmethod
    def parse(cls, string):
        ind = string.index('{')
        return __class__([DefinitionWithBaseAST.parse(string[:ind]),
                          BodyAST.parse(string[ind:].strip())])


class DefinitionWithBaseAST(AST):
    @classmethod
    def parse(cls, string):
        children = list(map(str.strip, cls.split(string, end=')')))
        assert len(children) <= 2
        definition = DefinitionAST.parse(children[0])
        if len(children) == 1:
            return __class__([definition])
        else:
            return __class__([definition, DefinitionBaseAST.parse(children[1])])


class DefinitionAST(AST):
    @classmethod
    def parse(cls, string):
        ind = string.index('(')
        return __class__([*cls.get_function_def(string[:ind].strip()),
                          ParametersAST.parse(string[ind:])])

    @classmethod
    def get_function_def(cls, string):
        tps = []
        while string:
            tp, string = cls.get_type(string)
            string = string.strip()
            tps.append(tp)
        return map(DottedName, tps)


class DefinitionBaseAST(AST):
    @classmethod
    def parse(cls, string):
        return None # FIXME:
        assert string[0] == ':'
        return __class__([DefinitionAST.parse(string[1:])])


class ParametersAST(AST):
    @classmethod
    def parse(cls, string):
        return __class__(list(map(ParameterAST.parse, cls.split(string))))

    @staticmethod
    def split(string):
        string = string.strip()
        assert string[0] == '(' and string[-1] == ')'
        res = [""]
        opens = '({[<'
        closes = ')}]>'
        s = 0
        for i in string[1:-1]:
            if i in opens:
                s += 1
            if i in closes:
                s -= 1
            if s == 0 and i == ',':
                res.append("")
            else:
                res[-1] += i
        if not res[-1].strip():
            res.pop()
        return res


class ParameterAST(AST):
    @classmethod
    def parse(cls, string):
        # ParameterTypeName [ '=' Expression]
        if '=' in string:
            ind = string.index('=')
            return __class__([ParameterTypeName.parse(string[:ind].strip()),
                              ParameterValue.parse(string[ind:].strip())])
        return __class__([ParameterTypeName.parse(string)])

class ParameterTypeName(AST):
    @classmethod
    def parse(cls, string):
        # ['ref' | 'out'] Type Variable_name
        param = re.findall(cls.NAME_PATTERN, string.strip())[-1]
        tp = string[:-len(param)].strip()
        tps = [tp]
        if cls.startswith(tp, 'ref') or cls.startswith(tp, 'out'):
            tps = [tp[:3], tp[3:].strip()]
        return __class__(list(map(DottedName, [*tps, param])))


class ParameterValue(AST):
    @classmethod
    def parse(cls, string):
        assert string[0] == '='
        return __class__([ExpressionAST.parse(string[1:].strip())])


class BodyAST(AST):
    @classmethod
    def parse(cls, string):
        assert string[0] == '{' and string[-1] == '}'
        string = string[1:-1].strip()
        return __class__([StatementsAST.parse(string)])


class StatementsAST(AST):
    @classmethod
    def parse(cls, string):
        lst = []
        while string:
            stmt, string = cls.get_statement(string)
            string = string.strip()
            if stmt:
                lst.append(stmt)
        return __class__(lst)

    @classmethod
    def get_statement(cls, string):
        string = string.strip()
        labeled_statement = re.match(cls.NAME_PATTERN + ':', string)
        if labeled_statement:
            ind = labeled_statement.end()
            return Label(string[:ind]), string[ind:]
        if string[0] == ';':
            return EmptyStatement([]), string[1:].strip()
        if string[0] == '{':
            block, string = cls.split_first(string)
            return BodyAST.parse(block), string
        if cls.startswith(string, 'try'):
            try_block, string = cls.split_first(string)
            string = string.strip()
            catches = []
            while cls.startswith(string, 'catch'):
                catch_block, string = cls.split_first(string)
                catches.append(catch_block)
            finally_blocks = []
            if cls.startswith(string, 'finally'):
                finally_block, string = cls.split_first(string)
                finally_blocks.append(finally_block)
            stmt = StatementAST([TryStatement.parse(try_block),
                                 *map(CatchStatement.parse, catches),
                                 *map(FinallyStatement.parse, finally_blocks)])

            return stmt, string
        if cls.startswith(string, 'if'):
            string = string[2:].strip()
            expr, string = cls.split_first(string, end=')')
            string = string.strip()
            body, string = cls.get_statement(string)
            lst = [body]
            string = string.strip()
            if cls.startswith(string, 'else'):
                string = string[4:].strip()
                else_clause, string = cls.get_statement(string)
                lst.append(else_clause)
            stmt = IfStatement([ExpressionAST.parse(expr), *lst])
            return stmt, string
        if cls.startswith(string, 'switch'):
            expr, string = cls.split_first(string[6:].strip(), end=')')
            expr = ExpressionAST.parse(expr)
            body, string = cls.split_first(string)
            return SwitchStatement([expr, SwitchBodyAST.parse(body)]), string
        if string.startswith("break;"):
            return BreakStatement([]), string[6:]

        if string.startswith("continue;"):
            return ContinueStatement([]), string[9:]

        if cls.startswith(string, 'goto'):
            string = string[4:].strip()
            ind = string.index(';')
            name, string = string[:ind], string[ind + 1:]
            return GotoStatement(list(map(DottedName,
                                          name.split()))), string
        if cls.startswith(string, 'throw'):
            expr, string = ExpressionAST.get_expression(
                string[5:].strip())
            string = string.strip()
            assert string[0] == ';'
            return ThrowStatement([expr]), string[1:].strip()

        if cls.startswith(string, 'return'):
            if string[6:].strip()[0] == ';':
                return ReturnStatement([]), string[6:].strip()[1:].strip()
            expr, string = ExpressionAST.get_expression(
                string[6:].strip())
            string = string.strip()
            assert string[0] == ';'
            return ReturnStatement([expr]), string[1:].strip()

        if cls.startswith(string, 'while'):
            expr, string = cls.split_first(string[5:].strip(), end=')')
            expr = ExpressionAST.parse(expr)
            body, string = cls.get_statement(string)
            return WhileStatement([expr, body]), string

        if cls.startswith(string, 'for'):
            exprs, string = cls.split_first(string[3:].strip(), end=')')
            init, cond, iterator = exprs[1:-1].split(';')
            body, string = cls.get_statement(string)
            return ForStatement([StatementAST.parse(init + ";"),
                                 ExpressionAST.parse(cond),
                                 ExpressionAST.parse(iterator),
                                 body]), string

        if cls.startswith(string, 'foreach'):
            iteration, string = cls.split_first(string[7:].strip(), end=')')
            iteration = iteration[1:-1].strip()
            tp, iteration = cls.get_type(iteration)
            iteration = iteration.strip()
            lst = [DottedName(tp)]
            ind = re.match(cls.NAME_PATTERN, iteration).end()
            var, iteration = iteration[:ind], iteration[ind:].strip()
            lst.append(DottedName(var))
            assert cls.startswith(iteration, "in")
            iteration = iteration[2:].strip()
            body, string = cls.get_statement(string)
            return ForEachStatement(lst + [body]), string

        if cls.startswith(string, "do"):
            body, string = cls.get_statement(string[2:].strip())
            assert cls.startswith(string, "while")
            expr, string = cls.split_first(string[5:].strip(), end=')')
            expr = ExpressionAST.parse(expr)
            string = string.strip()
            assert string[0] == ';'
            return DoWhileStatement([body, expr]), string[1:].strip()

        if cls.startswith(string, "using"):
            # TODO: using var x = new X();
            string = string[5:].strip()
            assert string[0] == '('
            stmt, string = cls.split_first(string, end=')')
            stmt = cls.parse(stmt[1:-1].strip() + ";")
            string = string.strip()
            # TODO: only block allowed
            assert string[0] == '{'
            body, string = StatementsAST.get_statement(string)
            return UsingStatement([stmt, body]), string

        if cls.is_decl(string):
            tp, string = cls.get_type(string)
            tp = DottedName(tp)
            ended = False
            decls = []
            string = string.strip()
            while not ended:
                ind = re.match(cls.NAME_PATTERN, string).end()
                name, string = string[:ind], string[ind:].strip()
                name = DottedName(name)
                lst = [name]
                if string[0] == '=':
                    string = string[1:].strip()
                    if string[0] == '{':
                        init, string = cls.split_first(string)
                        lst.append(ArrayInitializerList.parse(init))
                    else:
                        expr, string = ExpressionAST.get_expression(string)
                        lst.append(expr)
                decls.append(DeclarationStatement(lst))
                string = string.strip()
                if string[0] == ';':
                    ended = True
                elif string[0] == ',':
                    ended = False
                else:
                    assert False
                string = string[1:].strip()
            return DeclarationsStatement([tp, *decls]), string
        expr, string = ExpressionAST.get_expression(string)
        string = string.strip()
        assert string[0] == ';'
        return expr, string[1:].strip()
        # TODO: checked, unchecked, lock, using, yield
        # TODO: embedded_statement_unsafe ?? Unknown


    @classmethod
    def is_decl(cls, string):
        tp, string = cls.get_type(string)
        if tp is None:
            return False
        string = string.strip()
        return re.match(cls.NAME_PATTERN, string) is not None


class StatementAST(StatementsAST):
    pass


class TryStatement(StatementAST):
    @classmethod
    def parse(cls, string):
        assert cls.startswith(string, 'try')
        return __class__([BodyAST.parse(string[3:].strip())])


class CatchStatement(StatementAST):
    @classmethod
    def parse(cls, string):
        assert cls.startswith(string, 'catch')
        lst = []
        string = string[5:].strip()
        if string[0] == '(':
            exc, string = cls.split_first(string, end=')')
            lst.extend(map(DottedName, exc.split()))
        string = string.strip()
        if cls.startswith(string, 'when'):
            expr, string = cls.split_first(string[4:].strip(), end=')')
            lst.append(ExpressionAST.parse(expr.strip()))
        return __class__(lst + [BodyAST.parse(string)])


class FinallyStatement(StatementAST):
    @classmethod
    def parse(cls, string):
        assert cls.startswith(string, 'finally')
        return __class__([BodyAST.parse(string[7:].strip())])

class IfStatement(StatementAST):
    pass

class SwitchStatement(StatementAST):
    pass

class BreakStatement(StatementAST):
    pass

class ContinueStatement(StatementAST):
    pass

class GotoStatement(StatementAST):
    pass

class ThrowStatement(StatementAST):
    pass

class ReturnStatement(StatementAST):
    pass

class SwitchBodyAST(AST):
    @classmethod
    def parse(cls, string):
        string = string.strip()
        assert string[0] == '{' and string[-1] == '}'
        stmts = []
        string = string[1:-1].strip()
        while string:
            if string.startswith('default'):
                string = string[7:].strip()
                assert string[0] == ':'
                string = string[1:].strip()
                stmts.append(DottedName('default'))
            else:
                assert string.startswith('case')
                string = string[4:].strip()
                ind = string.index(':')
                tpname, string = string[:ind], string[ind + 1:].strip()
                stmts.append(DottedName(tpname))

            while string and (not string.startswith('case') or string.startswith('default')):
                stmt, string = StatementsAST.get_statement(string)
                stmts.append(stmt)
                string = string.strip()
        return __class__(stmts)


class EmptyStatement(StatementAST):
    pass

class WhileStatement(StatementAST):
    pass

class DoWhileStatement(StatementAST):
    pass

class ForStatement(StatementAST):
    pass

class ForEachStatement(StatementAST):
    pass

class UsingStatement(StatementAST):
    pass

class DeclarationsStatement(StatementAST):
    pass

class DeclarationStatement(StatementAST):
    pass

class ArrayInitializerList(AST):
    @classmethod
    def parse(cls, string):
        assert string[0] == '{' and string[-1] == '}'
        string = string[1:-1].strip()
        exprs = []
        while string:
            if string[0] == '{':
                init, string = cls.split_first(string)
                exprs.append(ArrayInitializerList.parse(init.strip()))
            else:
                expr, string = ExpressionAST.get_expression(string)
                exprs.append(expr)
            string = string.strip()
        return __class__(exprs)

class ExpressionAST(AST):
    PREF_UNARY = ["++", "+", "--", "-", "!", "~"]
    POST_UNARY = ["++", "--"]
    BINARY_OPERATORS = ["??", "or", "and", "is", "as", "||", "&&", "^",
                        "&", "|", "+", "-", "*", "/", "%",
                        ">>", "<<", "?.", ".",
                        "==", "!=", ">=", "<=", ">", "<"]
    TERNARY_OPERATORS = [["?", ":"]] # Won't use, just for the record


    @classmethod
    def parse(cls, string):
        expr, string = cls.get_expression(string.strip())
        assert not string.strip()
        return expr

    @classmethod
    def get_expression(cls, string):
        string = string.strip()
        if cls.is_lambda(string):
            return cls.get_lambda(string)
        if cls.is_assignment(string):
            return cls.get_assignment(string)
        if string[0] == "(":
            expr, string = cls.split_first(string, end=')')
            expr = cls.parse(expr[1:-1])
            string = string.strip()
            if re.match(cls.CONSTANT, string) is not None:
                expr2, string = cls.get_expression(string)
                return __class__([expr, expr2]), string
        else:
            for pref_un in cls.PREF_UNARY:
                if string.startswith(pref_un):
                    l = len(pref_un)
                    op, string = Operator(pref_un), string[l:].strip()
                    expr, string = cls.get_expression(string)
                    return __class__([op, expr]), string
            ind = re.match(cls.CONSTANT, string).end()
            name, string = DottedName(string[:ind]), string[ind:]
            if name.children == ["new"]:
                return cls.get_new_obj(string.strip())
            expr = __class__([name])

        string = string.strip()
        if not string or string[0] in ',;:':
            return expr, string

        for post_un in cls.POST_UNARY:
            if string.startswith(post_un):
                l = len(post_un)
                op, string = Operator(post_un), string[l:].strip()
                if string and string[0] not in ',;:':
                    nexpr, string = cls.get_expression(string)
                    p = [nexpr]
                else:
                    p = []
                expr = __class__([expr, op, *p])
                return expr, string

        string = string.strip()
        if not string or string[0] in ',;:':
            return expr, string
        mems = []
        while string:
            string = string.strip()
            for binmem in ["?[", "["]:
                if string.startswith(binmem):
                    mem = Operator(binmem)
                    name, string = cls.split_first(string, end=']')
                    name = cls.parse(name[len(binmem): -1])
                    mems.extend([mem, name])
                    break
            else:
                if string.startswith("("):
                    mem = Operator("()")
                    name, string = cls.split_first(string, end=')')
                    params = []
                    name = name[1:-1].strip()
                    while name:
                        param, name = cls.get_expression(name)
                        params.append(param)
                        name = name.strip()
                        if name:
                            assert name[0] == ','
                            name = name[1:].strip()
                    mems.extend([mem, *params])
                else:
                    if string.startswith("<"):
                        mem = Operator("<")
                        string_ = string[1:].strip()
                        obj = re.match(cls.DOTTED_NAME_PATTERN, string_)
                        if obj is None:
                            break
                        ind = obj.end()
                        name = string_[:ind]
                        string_ = string_[ind:].strip()
                        if string_ and string_[0] == ">":
                            name = cls.parse(name)
                            mems.extend([mem, name])
                            string = string_[1:].strip()
                        else:
                            break
                    else:
                        break


        if not string or string[0] in ',;:':
            return __class__([expr, *mems]), string

        for binop in cls.BINARY_OPERATORS:
            if string.startswith(binop):
                op = Operator(binop)
                string = string[len(binop):].strip()
                operands = []
                if binop in ["is", "as"]:
                    sop, string = cls.get_type(string)
                    operands.append(DottedName(sop))
                    string = string.strip()
                    if binop == "is":
                        obj = re.match(cls.NAME_PATTERN, string)
                        if obj is not None:
                            ind = obj.end()
                            top, string = string[:ind], string[ind:].strip()
                            operands.append(DottedName(top))
                        else:
                            top, string = cls.get_expression(f"A {string}")
                            top = top.children[1:]
                            operands.extend(top)
                else:
                    sop, string = cls.get_expression(string)
                    operands.append(sop)
                return __class__([expr, *mems, op, *operands]), string

        if string[0] == "?":
            texpr, string = cls.get_expression(string[1:].strip())
            string = string.strip()
            assert string[0] == ":"
            fexpr, string = cls.get_expression(string[1:].strip())
            string = string.strip()
            return __class__([expr, texpr, fexpr]), string
        print('----', expr, '---', string[:200])
        assert False

    @classmethod
    def is_assignment(cls, string):
        name, s = cls.get_dotted_member_name(string)
        if name is None:
            return False
        return any(s.startswith(eq) for eq in ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=']) and not s.startswith('==')

    @classmethod
    def get_assignment(cls, string):
        assert cls.is_assignment(string)
        member, string = cls.get_dotted_member_name(string)
        ind = string.index('=')
        assign_op, string = string[:ind + 1].strip(), string[ind + 1:].strip()
        expr, string = cls.get_expression(string)
        return AssignmentAST([member, Operator(assign_op), expr]), string

    @classmethod
    def is_lambda(cls, string):
        if string[0] != '(':
            obj = re.match(cls.NAME_PATTERN, string)
            if obj is None:
                return False
            ind = obj.end()
            string = string[ind:].strip()
        else:
            _, string = cls.split_first(string, end=')')
        return string.strip().startswith('=>')

    @classmethod
    def get_lambda(cls, string):
        ind = string.index('=>')
        params, string = string[:ind].strip(), string[ind:].strip()
        assert string[:2] == '=>'
        params = LambdaParametersAST.parse(params)
        string = string[2:].strip()
        if string[0] == '{':
            body, string = StatementsAST.get_statement(string)
            return LambdaExpression([params, body]), string
        expr, string = cls.get_expression(string)
        return LambdaExpression([params, expr]), string

    @classmethod
    def get_new_obj(cls, string):
        tp, string_ = cls.get_type(string)
        if tp:
            string = string_.strip()
            tps = [DottedName(tp)]
        else:
            tps = []
        call = False
        params = []
        if string.startswith("("):
            call = True
            expr, string = cls.split_first(string, end=')')
            expr = expr[1:-1].strip()
            while expr:
                p, expr = cls.get_expression(expr)
                params.append(p)
                expr = expr.strip()
                assert not expr or expr[0] == ','
                expr = expr[1:].strip()
            string = string.strip()
        if string.startswith("{"):
            expr, string = cls.split_first(string, end='}')
            expr = expr[1:-1].strip()
            iskv = cls.is_kv(expr)
            if call:
                assert iskv
            while expr:
                if iskv:
                    ind = re.match(cls.NAME_PATTERN, expr).end()
                    params.append(DottedName(expr[:ind]))
                    expr = expr[ind:].strip()
                    assert expr[0] == '='
                    expr = expr[1:].strip()
                p, expr = cls.get_expression(expr)
                params.append(p)
                expr = expr.strip()
                assert not expr or expr[0] == ','
                expr = expr[1:].strip()
        obj = __class__([DottedName("new"), *tps, *params])
        return obj, string
    
    @classmethod
    def is_kv(cls, string):
        m = re.match(cls.NAME_PATTERN, string)
        if not m:
            return False
        ind = m.end()
        rem = string[ind:].strip()
        return len(rem) < 2 and rem == "=" or rem[0] == "=" and rem[1] != "="

class LambdaExpression(ExpressionAST):
    pass

class LambdaParametersAST(AST):
    @classmethod
    def parse(cls, string):
        string = string.strip()
        if not string[0] == '(':
            assert re.match(cls.NAME_PATTERN, string).group() == string
            return __class__([DottedName(string)])
        assert string[0] == '(' and string[-1] == ')'
        string = string[1:-1].strip()
        params = list(map(lambda p: ParametersAST(list(map(DottedName, map(str.strip, p.split())))), map(str.strip, string.split(','))))
        return __class__(params)

class Label(AST):
    def __init__(self, name):
        assert name[-1] == ':'
        super().__init__([DottedName(name[:-1].strip())])


class DottedName(AST):
    def __init__(self, name):
        if not isinstance(name, str):
            super().__init__(name)
            return
        name = name.strip()
        if not any(c in name for c in '.<['):
            super().__init__([name])
        else:
            obj = re.match(self.STRING, name)
            if obj:
                ind = obj.end()
                super().__init__([__class__([name[:ind]]),
                                  __class__(name[ind:])])
                return
            inddot = name.index('.') if "." in name else len(name)
            indmem = name.index('[') if '[' in name else len(name)
            indtp = name.index('<') if '<' in name else len(name)
            if inddot < min(indmem, indtp):
                super().__init__([__class__(name[:inddot]),
                                  __class__(name[inddot + 1:])])
            elif indmem < indtp:
                nm1, name = __class__(name[:indmem].strip()), name[indmem:]
                nm2, name = self.split_first(name, end=']')
                nm2 = __class__(nm2[1:-1])
                name = name.strip()
                name = __class__(f"A{name}").children[1:]
                super().__init__([nm1, nm2, *name])
            else:
                nm1, name = __class__(name[:indtp].strip()), name[indtp:]
                nm2, name = self.split_first(name, end='>', opensadd="<", closesadd=">")
                nm2 = __class__(nm2[1:-1])
                name = name.strip()
                name = __class__(f"A{name}").children[1:]
                super().__init__([nm1, nm2, *name])

    def __repr__(self, tab=0, cls_name=True):
        if len(self.children) >= 2:
            s = ""
            if cls_name:
                s = '  ' * tab + __class__.__name__ + '\n'
                tab += 1
            return s + '\n'.join(list(map(
                lambda c: c.__repr__(tab, cls_name=False),
                self.children
                )))
        else:
            return '  ' * tab + self.children[0]

class AssignmentAST(AST):
    pass

class Operator(AST):
    def __init__(self, name):
        self.name = name.strip()
        super().__init__([name.strip()])

    def __repr__(self, tab=0, **kwargs):
        return '  ' * tab + self.name
