// Independent evaluation of an exported DRD with bpmn.io tooling:
// dmn-moddle parses the XML, feelin (a FEEL interpreter) evaluates expressions and unary tests.
//
//   node evaluate.mjs model.dmn cases.json > out.json
//
// cases.json: array of objects mapping input variable names to values.
// Output: array of objects with the output of every decision table, in dependency order.

import { readFileSync } from "node:fs";
import { DmnModdle } from "dmn-moddle";
import { evaluate, unaryTest } from "feelin";

const [, , dmnPath, casesPath] = process.argv;
const moddle = new DmnModdle();
const { rootElement: defs, warnings } = await moddle.fromXML(readFileSync(dmnPath, "utf8"));
if (warnings.length) {
  console.error(JSON.stringify(warnings.map((w) => w.message)));
  process.exit(2);
}

const decisions = defs.drgElement.filter((e) => e.$type === "dmn:Decision" && e.decisionLogic?.$type === "dmn:DecisionTable");
const byId = Object.fromEntries(decisions.map((d) => [d.id, d]));
const ordered = [];
const visit = (d) => {
  if (ordered.includes(d)) return;
  for (const req of d.informationRequirement || []) {
    const ref = req.requiredDecision?.href?.slice(1);
    if (ref && byId[ref]) visit(byId[ref]);
  }
  ordered.push(d);
};
decisions.forEach(visit);

function value(res) {
  return res && typeof res === "object" && "value" in res ? res.value : res;
}

function evalTable(table, ctx) {
  const inputs = (table.input || []).map((i) => value(evaluate(i.inputExpression.text, ctx)));
  const hits = [];
  for (const rule of table.rule) {
    const ok = (rule.inputEntry || []).every((entry, k) => {
      const text = entry.text.trim();
      return text === "-" || value(unaryTest(text, { ...ctx, "?": inputs[k] })) === true;
    });
    if (ok) hits.push(rule);
    if (ok && table.hitPolicy === "FIRST") break;
  }
  if (table.hitPolicy === "UNIQUE" && hits.length > 1) throw new Error(`UNIQUE violated in ${table.id}`);
  return hits.length ? value(evaluate(hits[0].outputEntry[0].text, ctx)) : null;
}

const cases = JSON.parse(readFileSync(casesPath, "utf8"));
const out = cases.map((c) => {
  const ctx = { ...c };
  const res = {};
  for (const d of ordered) {
    const name = d.variable?.name ?? d.decisionLogic.output[0].name;
    ctx[name] = evalTable(d.decisionLogic, ctx);
    res[name] = ctx[name];
  }
  return res;
});
process.stdout.write(JSON.stringify(out));
