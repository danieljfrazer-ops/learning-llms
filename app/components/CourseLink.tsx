import type { AnchorHTMLAttributes } from 'react';

type CourseLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & { href: string };

/** Use durable document navigation while Vinext's beta client router is unstable. */
export default function CourseLink(props: CourseLinkProps) {
  return <a {...props} />;
}
