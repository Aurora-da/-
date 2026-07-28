from dataclasses import dataclass, field

@dataclass
class Group:
    name: str
    members: list = field(default_factory = list)

group1 = Group(name = 'Developers')
group2 = Group(name = 'Designers')

group1.members.append('Alice')
group2.members.append('Bob')

print(group1.members)
print(group2.members)