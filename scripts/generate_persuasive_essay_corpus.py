import attr
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, NewType, Union, List

AnnID = NewType('AnnID', str)


@attr.s(auto_attribs=True)
class AnnotationSpan:
    id: AnnID
    text: str
    span: Optional[Tuple[int, int]]


@attr.s(auto_attribs=True)
class MajorClaim(AnnotationSpan):
    claims: Optional[List[AnnID]] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class Claim(AnnotationSpan):
    major_claims:  Optional[List[AnnID]] = attr.ib(factory=list)
    stance: Optional[int] = None
    premises: Optional[List[AnnID]] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class Premise(AnnotationSpan):
    claim: Optional[AnnID] = None
    supporting_premises: Optional[List[AnnID]] = attr.ib(factory=list)


@attr.s(auto_attribs=True)
class EssayAnnotation:
    id: str = None
    major_claims: Dict[AnnID, MajorClaim] = attr.ib(factory=dict)
    claims: Dict[AnnID, Claim] = attr.ib(factory=dict)
    premises: Dict[AnnID, Premise] = attr.ib(factory=dict)

    @classmethod
    def pair_to_command(cls, pair: Tuple[List[str], str]) -> Tuple[str, str]:
        premises, claim = pair
        return f'summarize: {" ".join(premises)}', claim

    def make_premise_claim_pairs(self) -> List[Tuple[str, List[str], str]]:
        pairs = []
        for major_claim in self.major_claims.values():
            if major_claim.claims:
                supporting_claim_texts = [self.claims[claim_id].text for claim_id in major_claim.claims]
                pairs.append((self.id, supporting_claim_texts, major_claim.text))
        for claim in self.claims.values():
            if claim.premises:
                premise_texts = [self.premises[premise_id].text for premise_id in claim.premises]
                pairs.append((self.id, premise_texts, claim.text))
        for premise in self.premises.values():
            if premise.supporting_premises:
                supporting_premises_texts = [self.premises[premise_id].text for
                                             premise_id in premise.supporting_premises]
                pairs.append((self.id, supporting_premises_texts, premise.text))
        return pairs


def process_annotation_line(line: str, essay: EssayAnnotation) -> Union[MajorClaim, Claim, Premise]:
    ann_id_, ann_type_span, ann_text = line.split('\t')
    ann_type, ann_span_start, ann_span_end = ann_type_span.split()
    span = (int(ann_span_start), int(ann_span_end))
    ann_id = AnnID(ann_id_)
    if ann_type == 'MajorClaim':
        annotation = MajorClaim(id=ann_id, text=ann_text, span=span)
        essay.major_claims[ann_id] = annotation
    elif ann_type == 'Claim':
        annotation = Claim(id=ann_id, text=ann_text, span=span)
        essay.claims[ann_id] = annotation
    elif ann_type == 'Premise':
        annotation = Premise(id=ann_id, text=ann_text, span=span)
        essay.premises[ann_id] = annotation
    else:
        raise ValueError(f'AnnType is expected to be in ["MajorClaim", "Claim", "Premise"], got {ann_type}')
    return annotation


def process_stance_line(line: str, essay: EssayAnnotation):
    stance_id_, stance_info = line.split('\t')
    stance_type, claim_id_, stance = stance_info.split()
    claim_id = AnnID(claim_id_)
    assert claim_id in essay.claims, \
        f'claim {claim_id} not found in claims {essay.claims.items()}'
    if stance == 'For':
        essay.claims[claim_id].stance = 1
    else:
        essay.claims[claim_id].stance = 0


def process_rel_line(line: str, essay: EssayAnnotation):
    rel_id_, rel_info = line.split('\t')
    rel_type, premise_id_, claim_id_ = rel_info.split()
    premise_id = AnnID(premise_id_.split(':')[1])  # Arg1:T1
    claim_id = AnnID(claim_id_.split(':')[1])  # Arg2:T4
    assert claim_id in essay.claims or claim_id in essay.premises, \
        f'{claim_id} not found in claims {essay.claims.items()} or premises {essay.premises.items()} '
    assert premise_id in essay.premises, \
        f'premise {premise_id} not found in claims {essay.premises.items()}'
    if rel_type == 'supports':
        if claim_id in essay.claims:
            essay.premises[premise_id].claim = claim_id
            essay.claims[claim_id].premises.append(premise_id)
        elif claim_id in essay.premises:
            essay.premises[premise_id].claim = claim_id
            essay.premises[claim_id].supporting_premises.append(premise_id)


def parse_ann_file(name, text: str) -> EssayAnnotation:
    essay = EssayAnnotation(id=name)
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line:
            if line.startswith('T'):
                process_annotation_line(line, essay)
            elif line.startswith('A'):
                process_stance_line(line, essay)
            elif line.startswith('R'):
                process_rel_line(line, essay)
            else:
                raise ValueError(f'unexpected line parse at line {line}')

    for claim_id in essay.claims:  # add claims to support major claims
        if essay.claims[claim_id].stance == 1:  # only add supporting claims
            essay.claims[claim_id].major_claims = essay.major_claims.values()
            for major_claim_id in essay.major_claims:
                essay.major_claims[major_claim_id].claims.append(claim_id)

    return essay


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ann_dir', type=lambda x: Path(x))
    arg_parser.add_argument('--out_path', type=lambda x: Path(x))
    args = arg_parser.parse_args()

    pairs = []
    for root, dirs, files in os.walk(args.ann_dir):
        for name in files:
            if name.endswith('.ann'):
                with open(os.path.join(root, name)) as f:
                    text = f.read()
                essay = parse_ann_file(name.strip('.ann'), text)
                pairs += essay.make_premise_claim_pairs()

    with open(args.out_path, 'w') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(('Essay', 'Premises', 'Claim'))
        for pair in pairs:
            writer.writerow((pair[0], ' ### '.join(pair[1]), pair[2]))


if __name__ == '__main__':
    main()
